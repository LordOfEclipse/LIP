function prox=shrink(x,gamma)
    
prox = sign(x).*max(abs(x) - gamma, 0);

end