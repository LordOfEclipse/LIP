function [x,error,time,i] = ISTA_BT(b,options,func_hand)
% *************************************
% ISTA_BT 
% *************************************
% F(x)=||A*x-b||_2+Lambda*||W*x||_1
% Q(x,y)=||A*y-b||_2+<x-y,grad(f(y))>+L/2*||x-y||_2^2+Lambda*||W*x||_1
% *************************************
% x_0=0,r_0=b
% for k
%   x_(k+1)=W^(-1)*(T_(Lambda*t)(W*(x_(k)+2*t*A'*r_(k))))
%   r_(k+1)=b-A*x_(k+1)
%   if F(x_(k+1))<Q(x_(k+1),x_(k))
%       break;
%   else
%       L=L*eta
%   end
% end
% *************************************

%% parameters

dim=options.dim;
max_outter_it=options.max_outter_it;
max_inner_it=options.max_inner_it;
Lambda=options.Lambda;
tol=options.tol;
error_upper_bound=options.error_upper_bound;
time_upper_bound=options.time_upper_bound;
display_in_figure=options.display_in_figure;
eta=options.eta;

x=zeros(dim);
if display_in_figure == true
    error=zeros(1,max_outter_it);
    time=zeros(1,max_outter_it);
end

%% funcs

F=func_hand.F;
Q=func_hand.Q;
proxR=func_hand.proxR;
dF=func_hand.dF;

%% iterations

timer=tic;
for i = 1 : max_outter_it
    
    L=options.L;
    for j = 1 : max_inner_it
        switch options.W_config
            case 'None'
                x_new=proxR((x-2/L*dF(x)),Lambda/L);
        end

        if F(x_new)<=Q(x_new,x,L)
            break;
        else
            L=L*eta;
        end
    end
    
    err=val_error(x_new,x,b,options,func_hand);

    if display_in_figure == true
        error(i)=err;
    end

    x=x_new;

    ti=toc(timer);

    if display_in_figure == true
        time(i)=ti;
    end

    if err<tol||err>error_upper_bound||ti>time_upper_bound
        break;
    end
    
end

if display_in_figure == true
    error=error(1:i);
    time=time(1:i);
else
    error=err;
    time=ti;
end

end