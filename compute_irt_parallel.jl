using Pkg
Pkg.activate(".")
using CSV
using DataFrames
using DynamicAxisWarping
using Distances
using Glob
using Statistics
using ArgParse
using Base.Threads

const SAMPLING_RATE = 30  # Hz — sampling rate for timestamp conversion

# Function to forward fill NaN values
function ffill!(vec)
	for k in 1:length(vec)
		if isnan(vec[k]) && k > 1
			vec[k] = vec[k-1]
		end
	end
end

# Function to load and preprocess CSV data
function load_cpCST_csv(filepath)
	fr = CSV.read(filepath, DataFrame)
	ffill!(fr.user_pos)
	ffill!(fr.stim_pos)
	fr[!, :user_pos] = fr.user_pos * -1
	fr[!, :time_secs] = fr.flip_time .- fr.flip_time[1]
	return fr
end

# Function to compute DTW values
function get_dtw_vals(DF, thresh=1e-12, radius=120)
	a = DF.stim_pos
	b = DF.user_pos
	dist = SqEuclidean(thresh)
	cost, i1, i2 = fastdtw(a, b, dist, radius)
	dat = DataFrame(user=i1, stim=i2)
	return dat
end

# Function to compute inter-response time and update DataFrame
function compute_irt!(DF)
	dat = get_dtw_vals(DF)
	n_vals = length(DF.user_pos)
	user_irt = zeros(n_vals)

	grouped = Dict{Int, Vector{Int}}()
	users = dat.user
	stims = dat.stim
	for i in eachindex(users)
		u = users[i]
		if !haskey(grouped, u)
			grouped[u] = Int[]
		end
		push!(grouped[u], stims[i])
	end

	inv_rate = 1 / SAMPLING_RATE
	for u in 1:n_vals
		if haskey(grouped, u)
			stim_indices = grouped[u]
			user_irt[u] = mean(stim_indices) * inv_rate - u * inv_rate
		end
	end

	DF[!, :irt] = user_irt
end

function process_files(source_folder, destination_folder)
	mkpath(destination_folder)
	csv_files = glob("*.csv", source_folder)
	println("Processing $(length(csv_files)) files...")

	Threads.@threads for file in csv_files
		try
			df = load_cpCST_csv(file)
			compute_irt!(df)
			dest_file = joinpath(destination_folder, basename(file))
			CSV.write(dest_file, df)
		catch e
			@warn "Failed to process $file" exception=(e, catch_backtrace())
		end
	end
end

parser = ArgParseSettings()

@add_arg_table parser begin
	"source_folder"
	help = "Path to the source folder containing CSV files"
	
	"destination_folder"
	help = "Path to the destination folder to save processed CSV files"
end

# Parse command line arguments
parsed_args = parse_args(parser)

source_folder = parsed_args["source_folder"]
destination_folder = parsed_args["destination_folder"]
process_files(source_folder, destination_folder)
