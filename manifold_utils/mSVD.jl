function eigen_calc_from_dist_mat(points::Array{Float64, 2}, center_ind::Int64, dist_mat::Array{Float64, 2}, Rmin::Float64, Rmax::Float64, radint::Float64, k::Int64)
	# Get number of points and ambient dimension
	N, d = size(points)

	# Get radii of interest
	radii = range(Rmin, stop=Rmax, step=radint)

	# Get distances from center point
	dist_vec = dist_mat[center_ind, :]

	# Sort indices by distance from center point
	sorted_indices = sort(1:N, by = x -> dist_vec[x])

	# Create array of radius-indices Tuples, where
	# "indices" is of type Array{Int64, 1}
	tuple_iterable = Array{Tuple{Float64, Array{Int64, 1}}, 1}()
	dist_ind::Int64 = 1
	for r in radii
		new_inds = Array{Int64, 1}()
		while dist_vec[sorted_indices[j]] <= r
			push!(new_inds, sorted_indices[j]
			dist_ind += 1
		end
		push!(tuple_iterable, (r, new_inds))
	end
end
