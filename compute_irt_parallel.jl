# Import necessary packages
using Pkg
Pkg.add("ArgParse")
Pkg.add("SentinelArrays")
Pkg.activate(".")
using CSV
using DataFrames
# using CairoMakie
using DynamicAxisWarping
using Distances
using Glob
using Smoothers
using Query
using SentinelArrays
using Statistics
using ArgParse
using Base.Threads
# Activate CairoMakie
# CairoMakie.activate!()

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

# Function to get a specific row based on target value
function get_row(dat, tgt_val)
	tmp = @from i in dat begin
		@where i.user == tgt_val
		@select {i.user, i.stim}
		@collect DataFrame
	end
	tmp[!, :user_ts] = tmp.user .* (1/60)
	tmp[!, :stim_ts] = tmp.stim .* (1/60)
	return tmp
end

# Function to compute pointwise reaction time
function get_point_rt(dat)
	idx = dat.user[1]
	user_ts = dat.user_ts[1]
	mean_stim = mean(dat.stim_ts)
	inst_rt = mean_stim - user_ts
	return inst_rt
end

# Function to compute inter-response time and update DataFrame
function compute_irt!(DF)
	dat = get_dtw_vals(DF)
	n_vals = length(DF.user_pos)
	user_irt = zeros(n_vals)
	# dtw_stim = zeros(n_vals)  # New array for DTW-shifted stimulus positions
	# dtw_user = zeros(n_vals)  # New array for DTW-shifted stimulus positions
	
	for u in 1:n_vals
		rdat = get_row(dat, u)
		irt = get_point_rt(rdat)
		user_irt[u] = irt
		# dtw_stim[u] = mean(rdat.stim_ts) * 60  # Convert back to position units
		# dtw_user[u] = mean(rdat.user_ts) * 60  # Convert back to position units
	end
	
	DF[!, :irt] = user_irt
	# DF[!, :dtw_stim] = dat.stim  # Add new column for DTW-shifted stimulus positions
	# DF[!, :dtw_user] = dat.user  # Add new column for DTW-shifted stimulus positions
end

# Main script execution
function process_files(source_folder, destination_folder)
	# Get all CSV files in the source folder
	csv_files = glob("*.csv", source_folder)
	
	# Process files in parallel
	Threads.@threads for file in csv_files
		# Load and process each CSV file
		df = load_cpCST_csv(file)
		compute_irt!(df)
		
		# Construct the destination file path
		dest_file = joinpath(destination_folder, basename(file))
		
		# Save the processed DataFrame to the destination folder
		CSV.write(dest_file, df)
	end
end

# Example usage
using ArgParse

# Create a parser object
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
