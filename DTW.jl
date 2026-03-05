### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ bdb713b2-e0aa-11ef-0495-1773f74653d8
begin
    using Pkg
    Pkg.add("ArgParse")
    Pkg.add("SentinelArrays")
    Pkg.activate(".")
    using CSV
    using DataFrames
    using DynamicAxisWarping
    using Distances
    using Glob
    using Smoothers
    using Query
    using SentinelArrays
    using Statistics
    using ArgParse
    using Base.Threads
end

# ╔═╡ c57f51d1-6133-460c-8ad4-31d8ac0d8349
begin
    function ffill!(vec)
        for k in 1:length(vec)
            if isnan(vec[k]) && k > 1
                vec[k] = vec[k-1]
            end
        end
    end

    function load_cpCST_csv(filepath)
        fr = CSV.read(filepath, DataFrame)
        ffill!(fr.user_pos)
        ffill!(fr.stim_pos)
        fr[!, :user_pos] = fr.user_pos * -1
        fr[!, :time_secs] = fr.flip_time .- fr.flip_time[1]
        return fr
    end

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

    function get_point_rt(dat)
        idx = dat.user[1]
        user_ts = dat.user_ts[1]
        mean_stim = mean(dat.stim_ts)
        inst_rt = mean_stim - user_ts
        return inst_rt
    end

    function compute_irt!(DF, radius)
        dat = get_dtw_vals(DF, radius)
        n_vals = length(DF.user_pos)
        user_irt = zeros(n_vals)

        for u in 1:n_vals
            rdat = get_row(dat, u)
            irt = get_point_rt(rdat)
            user_irt[u] = irt
        end

        DF[!, :irt] = user_irt
    end

    function process_file_thread_A(file, destination_folder)
        df = load_cpCST_csv(file)
        radius = 120
        compute_irt!(df, radius)

        mean_irt = mean(df.irt)
        println("Processing file: ", file)
        println("Initial IRT mean: ", mean_irt, " | Radius: ", radius)

        if mean_irt > 3 || mean_irt < 0
            println("IRT out of range, switching to Thread B")
            process_file_thread_B(file, destination_folder, 110)
        else
            dest_file = joinpath(destination_folder, basename(file))
            CSV.write(dest_file, df)
            println("File processed and saved successfully: ", dest_file)
        end
    end

    function get_dtw_vals(DF, radius, thresh=1e-12)
        a = DF.stim_pos
        b = DF.user_pos
        dist = SqEuclidean(thresh)
        cost, i1, i2 = fastdtw(a, b, dist, radius)
        dat = DataFrame(user=i1, stim=i2)
        return dat
    end

    function process_file_thread_B(file, destination_folder, radius, iteration=1)
        df = load_cpCST_csv(file)
        compute_irt!(df, radius)

        while (mean(df.irt) > 3 || mean(df.irt) < 0) && iteration <= 6
            println("Iteration: ", iteration, " | File: ", file, " | Radius: ", radius, " | Mean IRT: ", mean(df.irt))
            radius -= 10
            iteration += 1
            compute_irt!(df, radius)
        end

        dest_file = joinpath(destination_folder, basename(file))
        CSV.write(dest_file, df)
        println("Final processed file saved: ", dest_file, " | Final Radius: ", radius)
    end

    function process_files(source_folder, destination_folder)
        csv_files = glob("*.csv", source_folder)

        Threads.@threads for file in csv_files
            process_file_thread_A(file, destination_folder)
        end
    end
end

# ╔═╡ debc58f9-b213-49cf-baa1-4ee105c572e0
begin
    source_folder = "/Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/IRT_extraction/test_data/feb_1_2025/test_4/proc_cpCST_data"
    destination_folder = "/Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/IRT_extraction/test_data/feb_1_2025/test_4/irt_data"
    process_files(source_folder, destination_folder)
end

# ╔═╡ Cell order:
# ╠═bdb713b2-e0aa-11ef-0495-1773f74653d8
# ╠═c57f51d1-6133-460c-8ad4-31d8ac0d8349
# ╠═debc58f9-b213-49cf-baa1-4ee105c572e0