using AbstractAlgebra
using LinearAlgebra
using NPZ

struct Z2_element <: FinFieldElem
	value::Bool
end

function +(x::Z2_element, y::Z2_element)::Z2_element
	return Z2_element(xor(x.value, y.value))
end

function -(x::Z2_element, y::Z2_element)::Z2_elements
	return Z2_element(xor(x.value, y.value))
end

function *(x::Z2_element, y::Z2_element)::Z2_elements
	return Z2_element(x.value & y.value)
end

#zero(::Type{Z2_element}) = Z2_element(false)
#one(::Type{Z2_element}) = Z2_element(true)
zero(x::Z2_element) = Z2_element(false)
zero(::Z2_element) = Z2_element(false)
Z2_element() = Z2_element(false)
#zero(::Type{Z2_element}) = Z2_element(false)

# Read in NumPy boundary map
edge_to_vert = npzread("edge_to_vert.npy")
n_verts, n_edges = size(edge_to_vert)

# Build boundary map with our data type
boundary_map = Array{Z2_element}(undef, n_verts, n_edges)
for j = 1:n_verts
	for k = 1:n_edges
		boundary_map[j, k] = Z2_element(edge_to_vert[j, k])
	end
end

bool_svd = svd(boundary_map)
