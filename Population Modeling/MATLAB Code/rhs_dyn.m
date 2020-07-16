function rhs = rhs_dyn(t,x,b,p,r,d)
rhs = diag([b - p*x(2,:), r*x(1,:) - d])*x;