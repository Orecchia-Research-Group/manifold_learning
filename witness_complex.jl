using Combinatorics

function weak_witness_complex(witness_dist_mat, max_deg::Int64)
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
		sorted_col = sort(witness_dist_mat[:, j])
		push!(edges, Tuple(sorted_col[1:2]))
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
				push!(new_simplices, intersection(combo))
			end
		end
		simplices[j] = new_simplices
	end

	return simplices
end

arr = [1 2 3;
	3 4 5;
	5 6 7;
	8 9 10]

max_deg = 3
simplices = weak_witness_complex(arr, max_deg)
for j in 0:max_deg
	println("Simplices of degree "*string(j)*":")
	current_simplices = simplices[j]
	for x in current_simplices
		println(x)
	end
end
