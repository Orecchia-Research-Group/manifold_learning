using NPZ
using Combinatorics
using ProgressBars

function weak_witness_complex_simplices(witness_dist_mat, max_deg::Int64)
	# n is the number of landmarks
	# N is the number of non-landmark points
	n, N = size(witness_dist_mat)

	# Store simplices in a set
	# Start by storing vertices
	simplices = Dict{Int64, Set}()
	simplices[0] = Set{Int64}(1:n)

	# Iterate over non-landmarks, computing which
	# edge is witnessed by each non-landmark
	edges = Set{Tuple}()
	for j in 1:N
		column = witness_dist_mat[:, j]
		closest_landmarks = sort(collect(1:n), by=x->column[x])
		push!(edges, Tuple(closest_landmarks[1:2]))
	end
	simplices[1] = edges

	# Insert simplices of degree 2 up to max-degree
	for j in 2:max_deg
		new_simplices = Set{Tuple}()
		facets = simplices[j-1]
		for combo in combinations(1:n, j+1)
			cond = true
			for facet in combinations(combo, j)
				if !in(facet, facets)
					cond = false
					break
				end
			end
			if cond
				push!(new_simplices, combo)
			end
		end
		simplices[j] = new_simplices
	end

	return simplices
end

function simplices_to_npy(simplices::Dict{Int64, Set}, max_deg::Int64, file_prefix::String)
	for j in 0:max_deg
		to_transcribe = collect(simplices[j])
		len_transcription = length(to_transcribe)
		to_write = zeros(Int64, (len_transcription, j+1))
		for k in 1:len_transcription
			for kk in 1:(j+1)
				to_write[k, kk] = to_transcribe[k][kk]
			end
		end
		npzwrite(file_prefix*"_"*string(j)*".npy", to_write)
	end
end

# ks = (1, 5, 10, 50, 100, 500, 1000, 5000, 10000)
ks = (5,)
ps = (0.01, 0.05, 0.1, 0.2)
pps = (0.05, 0.1, 0.2)

for p in ps
	for pp in ProgressBar(pps)
		for k in ks
			toople = (k, p, pp)
			arr = npzread("data/sample_witness_slice_kis"*string(toople)*".npy")

			max_deg = 3
			simplices = weak_witness_complex_simplices(arr, max_deg)

			simplices_to_npy(simplices, max_deg, "data/sample_weak_witness_kis"*string(toople))
		end
	end
end
