using LinearAlgebra
using NPZ

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
		while dist_vec[sorted_indices[dist_ind]] <= r
			push!(new_inds, sorted_indices[dist_ind])
			dist_ind += 1
		end
		push!(tuple_iterable, (r, new_inds))
	end

	# Begin computing mSVD
	points_gradual = Array{Float64}(undef, 0, 0)
	Ss = Array{Array{Float64, 1}, 1}()
	Vts = Array{Array{Float64, 2}, 1}()

	for (r, indices) in tuple_iterable
		if !isempty(indices)
			if isempty(points_gradual)
				points_gradual = points[indices, :]
			else
				points_gradual = cat(points_gradual, points[indices, :], dims=1)
			end
			spec_svd = svd(points_gradual)
			push!(Ss, spec_svd.S)
			push!(Vts, spec_svd.Vt)
		else
			push!(Ss, last(Ss))
			push!(Vts, last(Vts))
		end
	end
end

points = npzread("data/phat_sct_var.npy")
dist_mat = npzread("data/phat_dist_mat.npy")
indices = npzread("data/sample_landmark_indices.npy")

#center_ind = indices[1]
Rmin = 30.0
Rmax = 40.0
radint = 0.1
k = 5

for ind in indices[1:10]
	@time eigen_calc_from_dist_mat(points, ind, dist_mat, Rmin, Rmax, radint, k)
end
