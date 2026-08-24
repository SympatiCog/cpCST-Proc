# Instantaneous reaction time (iRT) from stimulus/user tracking, via FastDTW.
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

const REQUIRED_COLS = ["flip_time", "stim_pos", "user_pos"]

# Forward fill NaN values
function ffill!(vec)
	for k in 1:length(vec)
		if isnan(vec[k]) && k > 1
			vec[k] = vec[k-1]
		end
	end
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
    compute_irt!(DF; radius=DTW_RADIUS)

Stimulus-anchored instantaneous reaction time: for each stimulus frame, the
mean timestamp of the user frames the warp path aligns to it, minus that
stimulus frame's own timestamp. Positive means the user lagged the stimulus.

Three things changed here.

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
	stim = DF.stim_pos
	user = DF.user_pos
	t = DF.flip_time
	n = length(stim)

	# Below the band half-width the constraint is meaningless and the warp path
	# collapses onto the diagonal, which yields iRT == 0 for every sample -- a
	# value that looks like a measurement and is not one. Refuse instead.
	n >= radius || error("$n samples is shorter than the DTW band ($radius); " *
	                     "iRT is not defined for this recording")

	# genuine Sakoe-Chiba limits; see the note on DTW_RADIUS for why this is
	# `dtw` with explicit bounds rather than `fastdtw`
	i2min, i2max = radiuslimits(radius, n, length(user))
	_, stim_idx, user_idx = dtw(stim, user, SqEuclidean(1e-12), i2min, i2max)

	sums = zeros(Float64, n)
	counts = zeros(Int, n)
	@inbounds for k in eachindex(stim_idx)
		s = stim_idx[k]
		sums[s] += t[user_idx[k]]
		counts[s] += 1
	end

	irt = Vector{Float64}(undef, n)
	@inbounds for s in 1:n
		# a stimulus frame absent from the path has no defined iRT; the old
		# code indexed an empty frame and threw
		irt[s] = counts[s] == 0 ? NaN : sums[s] / counts[s] - t[s]
	end

	DF[!, :irt] = irt
	DF[!, :dtw_radius] = fill(radius, n)
	return DF
end

function process_files(source_folder, destination_folder; radius::Int=DTW_RADIUS)
	mkpath(destination_folder)
	csv_files = glob("*.csv", source_folder)
	failures = Threads.Atomic{Int}(0)

	Threads.@threads for file in csv_files
		# isolate per file: the folder also holds LSL marker CSVs with an
		# entirely different schema, and one of them used to kill the run
		try
			df = load_cpCST_csv(file)
			compute_irt!(df; radius=radius)
			CSV.write(joinpath(destination_folder, basename(file)), df)
		catch err
			Threads.atomic_add!(failures, 1)
			@warn "skipped" file exception=(err, catch_backtrace())
		end
	end

	n_ok = length(csv_files) - failures[]
	println("processed $(n_ok)/$(length(csv_files)) files at radius $radius")
	return n_ok
end

parser = ArgParseSettings()
@add_arg_table parser begin
	"source_folder"
	help = "Path to the source folder containing CSV files"
	"destination_folder"
	help = "Path to the destination folder to save processed CSV files"
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
