using Combinatorics

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
				push!(new_simplices, intersection(combo))
			end
		end
		simplices[j] = new_simplices
	end

	return simplices
end

function array_lt(array1::Array{Int64, 1}, array2::Array{Int64, 1})
	# iterate over indices until array1 is not equal
	# to array2 at some index
	ind = 1
	while true
		item1 = array1[ind]
		item2 = array2[ind]
		if item1 == item2
			ind += 1
		else
			return item1 < item2
		end
	end
end

function simplices_to_Eirene_complex(simplices::Dict{Int64, Set}, max_deg::Int64)
	# Start by sorting j-simplices for k in 0:max_deg
	sorted_simplices = Dict{Int64, Array}()
	for j in 0:max_deg
		list_of_simplices = collect(simplices[j])
		if j == 0
			sorted_simplices[j] = sort(list_of_simplices)
		else
			sorted_simplices[j] = sort(list_of_simplices, lt=array_lt)
		end
	end

	# Put all simplices into single array
	simplex_array = Array{Union{Int64, Tuple}, 1}()
	for j in 0:max_deg
		for item in sorted_simplices[j]
			push!(simplex_array, item)
		end
	end

	# Store number of simplices
	n_simplices = length(simplex_array)

	# Map each simplex to its new array index
	simplex_inds = Dict{Union{Int64, Tuple}, Int64}()
	for j in 1:n_simplices
		println(simplex_array[j])
		simplex_inds[simplex_array[j]] = j
	end

	# Form "codimension-1" matrix
	D = zeros(Int8, n_simplices, n_simplices)
	for j in n_simplices
		simplex = simplex_array[j]
		if isa(simplex, Tuple)
			len_simplex = length(simplex)
			facets = combinations(simplex, len_simplex-1)
			# Account for when 1-simplices has vertex-boundaries
			if len_simplex == 2
				for facet in facets
					k = simplex_inds[facet[1]]
					D[j, k] = 1
				end
			# General case
			else
				for facet in facets
					k = simplex_inds[facet]
					D[j, k] = 1
				end
			end
		end
	end

	# Get S, rv, cp per Eirene documentation
	S = sparse(D)
	rv = S.rowval
	cp = S.colptr

	return D
end

arr = [1 2 3;
	3 4 5;
	5 6 7;
	8 9 10]

max_deg = 3
simplices = weak_witness_complex_simplices(arr, max_deg)

D = simplices_to_Eirene_complex(simplices, max_deg)

print(D)
