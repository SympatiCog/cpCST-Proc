### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ bdb713b2-e0aa-11ef-0495-1773f74653d8
md"""
# cpCST instantaneous reaction time

This notebook is now a thin front end over `compute_irt_parallel.jl`. It used
to carry its own copy of the pipeline, which is how the 30 Hz/60 Hz error came
to live in two places at once; there is one implementation now.

The adaptive-radius search that used to live here has been removed. It ran the
whole pipeline at radius 120 and, if the mean iRT fell outside [0, 3], re-ran
it at 110, 100, ... down to 60 until the mean landed in range. Narrowing the
Sakoe-Chiba band does not correct an out-of-range estimate, it truncates it, so
the search always terminated eventually and the radius it stopped at was a
property of the stopping rule rather than of the participant. It was also
testing against values that were half their true size, because of the sampling
rate error. Use the fixed `DTW_RADIUS`, and report it.
"""

# ╔═╡ c57f51d1-6133-460c-8ad4-31d8ac0d8349
include(joinpath(@__DIR__, "compute_irt_parallel.jl"))

# ╔═╡ 3f9a1c04-5d21-4b7e-9c88-2a4e6f1b0d77
begin
	source_folder = "./proc_cpCST_data"
	destination_folder = "./irt_data"
end

# ╔═╡ debc58f9-b213-49cf-baa1-4ee105c572e0
process_files(source_folder, destination_folder; radius=DTW_RADIUS)

# ╔═╡ Cell order:
# ╟─bdb713b2-e0aa-11ef-0495-1773f74653d8
# ╠═c57f51d1-6133-460c-8ad4-31d8ac0d8349
# ╠═3f9a1c04-5d21-4b7e-9c88-2a4e6f1b0d77
# ╠═debc58f9-b213-49cf-baa1-4ee105c572e0
