function norm = normQ(x,Q)

norm = sqrt(sum(x'*Q*x,'all'));

end

