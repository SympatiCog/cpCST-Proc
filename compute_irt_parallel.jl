# Instantaneous reaction time (iRT) from stimulus/user tracking, via banded DTW,
# aligned separately over each crash-free epoch.
#
# Environment: activate the project that sits next to this script, THEN add to
# it. The original added to the default environment and activated a different
# one afterwards, so the packages were not necessarily in the project that ran.
using Pkg
Pkg.activate(@__DIR__)
for pkg in ("CSV", "DataFrames", "DynamicAxisWarping", "Distances",
            "Glob", "Statistics", "ArgParse")
    try
        @eval import $(Symbol(pkg))
    catch
        Pkg.add(pkg)
    end
end

using CSV
using DataFrames
using DynamicAxisWarping
using Distances
using Glob
using Statistics
using ArgParse
using Base.Threads

# Sakoe-Chiba band half-width, in samples. At 30 Hz, 120 bounds the warp to
# 4.0 s -- comfortably past any plausible reaction time.
#
# The bound is only real because this uses `dtw` with explicit band limits.
# It was previously `fastdtw`, whose `radius` argument does NOT constrain the
# warp: FastDTW coarsens the series, aligns at low resolution, projects that
# path up and refines within `radius` cells OF THE PROJECTED PATH, not of the
# diagonal. A bad coarse alignment is inherited rather than corrected, so the
# final path can sit arbitrarily far off-diagonal. Measured on one continuous
# file at radius 120: offsets of -1584 and +1818 samples (-53 s and +61 s),
# with 61.6% of the path outside the nominal radius, producing iRT values down
# to -52.8 s. A genuine band caps the offset at exactly `radius`, removes every
# whole-file failure in the corpus, leaves well-behaved files unchanged to
# three decimals, and runs ~3.7x faster than FastDTW at this length.
#
# This used to be searched: start at 120 and shrink by 10 until the mean iRT
# fell inside [0, 3]. That loop was unsound twice over. It was tuning a knob
# that bounded nothing, and its effect was not even monotonic -- on one file
# the out-of-band fraction ran 63.1% at radius 120, 33.3% at 60, then 79.8% at
# 30. It was chasing a real failure with no instrument able to catch it.
# Estimates plateau by radius 60, so 120 leaves the upper tail unclipped.
const DTW_RADIUS = 120

# Samples blanked at each end of every aligned run. DTW's endpoint constraint
# pins the warp path to the corners, so the estimates there are artefacts.
# Measured extent is a single sample (MAE 0.34 s at the boundary, 0.006 s at
# the next, 0.000 s thereafter), so 3 is already generous. Do not inflate it
# "to be safe": the cost lands directly on coverage.
const EDGE_MASK = 3

const REQUIRED_COLS = ["flip_time", "stim_pos", "user_pos"]

# Tracking check. DTW always finds some path inside the band, so it returns
# plausible-looking iRT whether or not the participant was following the
# stimulus. iRT cannot flag a recording where they were not; this can.
#
# `track_corr` is the peak Pearson correlation of stimulus against the FLIPPED
# user position over user lags 0..TRACK_MAX_LAG_S, pooled over the valid runs
# (no pair spans a crash). Following the stimulus makes it positive. Measured
# 2026-10-06: CPT/CPTLITE median 0.82, minimum 0.655; Calibrate median 0.94,
# minimum 0.79. No file in the corpus falls below TRACK_MIN_CORR, so the flag
# is a guard for new data. It must stay on valid rows: the crash runaway is the
# user moving the wrong way at ~60x normal amplitude, and 2-3% of runaway rows
# is enough to drive a whole-recording correlation to -0.9.
const TRACK_MAX_LAG_S = 2.0
const TRACK_MIN_CORR = 0.5

# Forward fill NaN values
function ffill!(vec)
	for k in 1:length(vec)
		if isnan(vec[k]) && k > 1
			vec[k] = vec[k-1]
		end
	end
end

# pandas writes booleans as True/False. CSV.jl parses those as Bool by
# default, but be robust to a String column.
as_bool(col) = eltype(col) <: Bool ? Vector{Bool}(col) :
               [lowercase(string(x)) == "true" for x in col]

"Contiguous runs of `true`, as index ranges."
function valid_runs(valid::AbstractVector{Bool})
	runs = UnitRange{Int}[]
	start = 0
	for k in eachindex(valid)
		if valid[k] && start == 0
			start = k
		elseif !valid[k] && start != 0
			push!(runs, start:k-1)
			start = 0
		end
	end
	start != 0 && push!(runs, start:length(valid))
	return runs
end

# Load and preprocess CSV data
function load_cpCST_csv(filepath)
	fr = CSV.read(filepath, DataFrame)
	for c in REQUIRED_COLS
		hasproperty(fr, Symbol(c)) || error("missing column $c")
	end
	ffill!(fr.user_pos)
	ffill!(fr.stim_pos)
	fr[!, :user_pos] = fr.user_pos * -1
	fr[!, :time_secs] = fr.flip_time .- fr.flip_time[1]
	return fr
end

"""
    compute_irt!(DF; radius=DTW_RADIUS) -> skipped

Stimulus-anchored instantaneous reaction time: for each stimulus frame, the
mean timestamp of the user frames the warp path aligns to it, minus that
stimulus frame's own timestamp. Positive means the user lagged the stimulus.

Alignment runs separately over each contiguous run of `is_valid` samples (a
frame without that column is one all-valid run), so no warp path ever crosses
a crash. Samples outside a run, the `EDGE_MASK` samples at each end of a run,
and every sample of a run shorter than the band are NaN. Returns the number
of runs skipped for being shorter than the band; `n_epochs_aligned` records
the number aligned.

Three earlier changes worth keeping in mind.

1. Timestamps come from `flip_time` directly. The previous code multiplied the
   frame index by 1/60, but these data are sampled at 30 Hz (median frame
   interval 0.03333 s in 131 of 133 files), so every iRT it produced was
   exactly half its true value. Reading the clock removes the constant, and
   the assumption behind it, altogether.

2. One grouped pass over the warp path replaces a per-sample Query.jl scan of
   the whole alignment table. That scan was O(n^2), but measured cost was
   modest -- 0.35 s for a 17,894-sample CPT run against 0.07 s here, so 2-5x
   depending on length, not the order-of-magnitude I first assumed. The gain
   that matters more is the NaN guard below: the scan indexed row 1 of its
   result without checking it was non-empty.

3. Alignment is a properly banded `dtw` rather than `fastdtw`, whose radius
   argument never bounded the warp. See the note on `DTW_RADIUS`.

Note on orientation: `dtw(x, y, ...)` returns `i1` indexing `x` and `i2`
indexing `y` (verified against DynamicAxisWarping). Here `x` is the stimulus,
so `i1` is a stimulus index and `i2` a user index. The previous code stored
these in columns labelled the other way round and then subtracted against the
wrong labels, so the two errors cancelled and the arithmetic came out right.
Naming them correctly here keeps that from being "fixed" into a sign flip.
"""
function compute_irt!(DF; radius::Int=DTW_RADIUS)
	n = nrow(DF)
	valid = hasproperty(DF, :is_valid) ? as_bool(DF.is_valid) : trues(n)
	irt = fill(NaN, n)
	aligned = 0
	skipped = 0
	for rng in valid_runs(valid)
		# Below the band half-width the constraint is meaningless and the warp
		# path collapses onto the diagonal, which yields iRT == 0 for every
		# sample -- a value that looks like a measurement and is not one.
		if length(rng) < radius
			skipped += 1
			continue
		end
		align_run!(irt, DF.stim_pos, DF.user_pos, DF.flip_time, rng, radius)
		aligned += 1
	end
	aligned > 0 || error("no crash-free epoch of at least $radius samples; " *
	                     "iRT is not defined for this recording")

	r, lag = tracking_corr(DF.stim_pos, DF.user_pos, DF.flip_time, valid)
	DF[!, :irt] = irt
	DF[!, :dtw_radius] = fill(radius, n)
	DF[!, :n_epochs_aligned] = fill(aligned, n)
	DF[!, :track_corr] = fill(r, n)
	DF[!, :track_lag] = fill(lag, n)
	DF[!, :tracking_ok] = fill(r > TRACK_MIN_CORR, n)
	return skipped
end

"""
    tracking_corr(stim, user, t, valid; max_lag_s) -> (r, lag_s)

Peak correlation of `stim[i]` against `user[i + k]` over k = 0..max_lag, and
the lag in seconds at which it peaks. `user` must already be flipped (as the
loader leaves it). Pairs are drawn only from within a single valid run.
"""
function tracking_corr(stim, user, t, valid; max_lag_s=TRACK_MAX_LAG_S)
	dt = median(diff(t))
	K = round(Int, max_lag_s / dt)
	runs = valid_runs(valid)
	best_r, best_k = -Inf, 0
	for k in 0:K
		n = 0; sx = 0.0; sy = 0.0; sxx = 0.0; syy = 0.0; sxy = 0.0
		for rng in runs, i in first(rng):last(rng)-k
			x = stim[i]; y = user[i + k]
			n += 1; sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y
		end
		n > 2 || continue
		vx = sxx - sx*sx/n; vy = syy - sy*sy/n
		(vx > 0 && vy > 0) || continue
		r = (sxy - sx*sy/n) / sqrt(vx * vy)
		if r > best_r
			best_r, best_k = r, k
		end
	end
	isfinite(best_r) || return (NaN, NaN)
	return (best_r, best_k * dt)
end

function align_run!(irt, stim, user, t, rng, radius)
	n = length(rng)
	s = collect(view(stim, rng))
	u = collect(view(user, rng))
	tt = view(t, rng)

	# genuine Sakoe-Chiba limits; see the note on DTW_RADIUS for why this is
	# `dtw` with explicit bounds rather than `fastdtw`
	i2min, i2max = radiuslimits(radius, n, n)
	_, stim_idx, user_idx = dtw(s, u, SqEuclidean(1e-12), i2min, i2max)

	sums = zeros(Float64, n)
	counts = zeros(Int, n)
	@inbounds for k in eachindex(stim_idx)
		si = stim_idx[k]
		sums[si] += tt[user_idx[k]]
		counts[si] += 1
	end
	@inbounds for k in 1:n
		# a stimulus frame absent from the path has no defined iRT; the old
		# code indexed an empty frame and threw
		irt[rng[k]] = counts[k] == 0 ? NaN : sums[k] / counts[k] - tt[k]
	end
	for k in 1:min(EDGE_MASK, n)
		irt[rng[k]] = NaN
		irt[rng[end - k + 1]] = NaN
	end
	return irt
end

function process_files(source_folder, destination_folder; radius::Int=DTW_RADIUS)
	# Validate before doing anything. Omitting the destination used to reach
	# mkpath(nothing) and fail with a MethodError pointing at an internal line
	# rather than at the missing argument.
	source_folder === nothing && error("source_folder is required")
	destination_folder === nothing && error("destination_folder is required")
	isdir(source_folder) || error("source folder does not exist: $source_folder")
	radius > 0 || error("radius must be positive, got $radius")

	# stage 1 writes its crash event table into the same folder; it is not a
	# recording and would otherwise be skipped with a warning on every run
	csv_files = filter(f -> basename(f) != "crash_events.csv",
	                   glob("*.csv", source_folder))
	isempty(csv_files) && error("no CSV files found in $source_folder")
	mkpath(destination_folder)
	failures = Threads.Atomic{Int}(0)
	short_epochs = Threads.Atomic{Int}(0)
	not_tracking = Threads.Atomic{Int}(0)

	Threads.@threads for file in csv_files
		# isolate per file: the folder also holds LSL marker CSVs with an
		# entirely different schema, and one of them used to kill the run
		try
			df = load_cpCST_csv(file)
			Threads.atomic_add!(short_epochs, compute_irt!(df; radius=radius))
			if !df.tracking_ok[1]
				Threads.atomic_add!(not_tracking, 1)
				@warn "not tracking the stimulus; iRT is not interpretable" file track_corr=df.track_corr[1]
			end
			CSV.write(joinpath(destination_folder, basename(file)), df)
		catch err
			Threads.atomic_add!(failures, 1)
			@warn "skipped" file exception=(err, catch_backtrace())
		end
	end

	n_ok = length(csv_files) - failures[]
	println("processed $(n_ok)/$(length(csv_files)) files at radius $radius; " *
	        "$(short_epochs[]) epoch(s) shorter than the band left NaN; " *
	        "$(not_tracking[]) file(s) flagged tracking_ok = false")
	return n_ok
end

parser = ArgParseSettings()
@add_arg_table parser begin
	"source_folder"
	help = "Path to the source folder containing CSV files"
	required = true
	"destination_folder"
	help = "Path to the destination folder to save processed CSV files"
	required = true
	"--radius"
	help = "DTW Sakoe-Chiba band half-width in samples (4.0 s at 30 Hz)"
	arg_type = Int
	default = DTW_RADIUS
end

if abspath(PROGRAM_FILE) == @__FILE__
	parsed_args = parse_args(parser)
	process_files(parsed_args["source_folder"],
	              parsed_args["destination_folder"];
	              radius=parsed_args["radius"])
end
