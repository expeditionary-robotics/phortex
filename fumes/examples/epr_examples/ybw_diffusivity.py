"""Diffusivity Inclusion"""

"""
Analytical solution for a laminar plume without diffusion nor viscosity:
Equations:
u * grad (teta) = 0    Advection diffusion equation for scalar teta with no diffusion
grad * u = 0    Incompressibility
u * grad u + grad p = teta g z

b(z) width of plume
U(z) velox of plume
Teta(z) buoyancy in the center of plume and with no diffusion this is CONSTANT
Teta = - beta T   contribution to density due to temperature

Solution in 3D:
b = b_0 (z_0/z)^(1/4)
U = U_0(z/z_0)^(1/2) = (2 Teta_0 g z)^(1/2)

where:
U_0 = (2Teta_0 z_0 g)^(1/2)
z_0 = U_0^2/(2 g Teta_0)
Teta_0 = Teta (0)
U_0 = U (0)
b_0 = b (0)

Solution in 2D:
b = b_0 (z_0/z)
U = U_0(z/z_0)^(1/2) = (2 Teta_0 g z)^(1/2)  (the same as in 3D)

Buoyancy flux is constant, i.e.:
2D:  U*b = cost
3D: pi b^2 U = cost

Now if we want to know how the inclusion of diffusion affects the above
solutions at a height z above the source, we can make the following back
of the envelope calculation:

dz/dt = U =  (2 Teta_0 g z)^(1/2)
--> integrating in z and t -->
2(2Teta_0 g z)^(1/2) = t (time taken to go from source to height H)

Diffusivity acts on a length scale l_d= (k*t)^(1/2) where near the bottom
the ocean diapycnal diffusivity k can vary from 10^-5 m^2/s over a smooth
bottom to 10^-3 m^2/s over a rough bottom (Polzin et al. 1997).
I would use something in the middle, i.e. 10^-4 m^2/s

So you can calculate l_d as a function of z and if l_d(z) << b(z) then
it has negligible influence. If L_d(z) >> b(z) then the plume is all diffused out.



"""