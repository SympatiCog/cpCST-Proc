# Run from the repo root: julia --threads=auto tests/test_irt.jl
include(joinpath(@__DIR__, "..", "compute_irt_parallel.jl"))
using Test

const LAG = 5                      # user lags stimulus by 5 samples = 0.1667 s
function frame(n; lag=LAG)
	t = 3.5 .+ (0:n-1) ./ 30
	stim = sin.(2π .* t ./ 7)
	user = -circshift(stim, lag)   # raw sign; compute_irt! flips a copy internally
	DataFrame(flip_time=t, stim_pos=stim, user_pos=user)
end
pybool(v) = [x ? "True" : "False" for x in v]   # what pandas writes

mktempdir() do dir
	src = joinpath(dir, "in"); dst = joinpath(dir, "out"); mkpath(src)
	n = 600
	valid = trues(n); valid[200:260] .= false
	df = frame(n); df.is_valid = pybool(valid)
	CSV.write(joinpath(src, "gap.csv"), df)
	CSV.write(joinpath(src, "plain.csv"), frame(n))
	short = frame(n); sv = trues(n); sv[100:n] .= false; short.is_valid = pybool(sv)
	CSV.write(joinpath(src, "short.csv"), short)
	# moves with the stimulus's mirror image: raw user == +stim, so after the
	# loader's flip it is anti-correlated at every lag. Still aligned, but flagged.
	# (30 s period: the 7 s test sine would wrap back to positive within 2 s of lag)
	anti = frame(n); anti.stim_pos = sin.(2π .* anti.flip_time ./ 30)
	anti.user_pos = circshift(anti.stim_pos, LAG)
	CSV.write(joinpath(src, "anti.csv"), anti)

	n_ok = process_files(src, dst)
	@test n_ok == 3

	g = CSV.read(joinpath(dst, "gap.csv"), DataFrame)
	@test all(isnan, g.irt[200:260])
	@test all(isnan, g.irt[1:3]) && all(isnan, g.irt[197:199])
	@test all(isnan, g.irt[261:263]) && all(isnan, g.irt[n-2:n])
	fin = filter(!isnan, g.irt)
	@test length(fin) == n - 61 - 12
	@test abs(median(fin) - LAG/30) < 0.02
	@test all(==(2), g.n_epochs_aligned)
	@test all(==(DTW_RADIUS), g.dtw_radius)

	p = CSV.read(joinpath(dst, "plain.csv"), DataFrame)
	@test all(==(1), p.n_epochs_aligned)
	@test count(isnan, p.irt) == 6
	@test abs(median(filter(!isnan, p.irt)) - LAG/30) < 0.02

	@test !isfile(joinpath(dst, "short.csv"))

	# user_pos is flipped internally for alignment only; the output keeps the raw sign
	for (name, f) in (("gap.csv", g), ("plain.csv", p))
		src_df = CSV.read(joinpath(src, name), DataFrame)
		@test f.user_pos == src_df.user_pos
	end

	# tracking check: peak corr(stim, flipped user) over user lags 0..2 s
	for f in (g, p)
		@test all(>(0.99), f.track_corr)
		@test all(x -> abs(x - LAG/30) < 1e-9, f.track_lag)
		@test all(f.tracking_ok)
	end
	a = CSV.read(joinpath(dst, "anti.csv"), DataFrame)
	@test all(<(0), a.track_corr)
	@test !any(a.tracking_ok)
	@test count(!isnan, a.irt) > 0          # still aligned; the flag is the guard
	@test length(unique(a.track_corr)) == 1  # constant per file
end
println("test_irt.jl passed")
