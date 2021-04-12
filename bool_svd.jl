using NPZ

struct Z2_element
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

zero(::Type{Z2_element}) = Z2_element(false)

edge_to_vert = npzread("edge_to_vert.npy")
#println(typeof(edge_to_vert))

#boundary_map = zeros(Z2_element, size(edge_to_vert))
n_verts, n_edges = size(edge_to_vert)
boundary_map = Array{Z2_element}(undef, n_verts, n_edges)
