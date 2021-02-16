using HDF5
using Combinatorics

function union_iterable(iter_obj::Array)
	return_obj = Set{Tuple{Vararg{Int64}}}()
	union!(return_obj, iter_obj)
	return Tuple(return_obj)
end

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
		# Initialize j-simplices
		new_simplices = Set{Tuple}()
		# Get j-1 simplices
		facets = collect(simplices[j-1])
		# Iterate over all combinations of (j+1) facets
		for combo in combinations(facets, j+1)
			# Check that all facets share a facet
			cond = true
			for pair in combinations(combo, 2)
				if length(intersect(pair[1], pair[2])) != (j-1)
					cond = false
				end
			end
			if cond
				onion = union_iterable(combo)
				push!(new_simplices, onion)
			end
		end
		simplices[j] = new_simplices
	end

	return simplices
end

function simplices_to_HDF5(simplices::Dict{Int64, Set}, max_deg::Int64, filename::String)
	touch(filename)
	fid = h5open(filename, "w")
	for j in 0:max_deg
		write(fid, string(j), string(collect(simplices[j])))
	end
	close(fid)
end

arr = h5read("data/sample_witness_slice.h5", "witness_slice")

#arr = [1 2 3 4 5;
#	0.5 1.5 2.5 3.5 4.5]

#max_deg = 3
max_deg = 2
@time simplices = weak_witness_complex_simplices(arr, max_deg)

simplices_to_HDF5(simplices, max_deg, "data/sample_weak_witness.h5")
