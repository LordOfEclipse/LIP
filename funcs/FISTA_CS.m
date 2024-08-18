function [x,error,time,i] = FISTA_CS(b,options,func_hand)
% *************************************
% FISTA_CS 
% *************************************
% x_0=0,y_0=0,r_0=b,t1=1,t2=1
% for k
%   x_(k+1)=y^(-1)*(T_(Lambda*t)(y*(y_(k)+2*t*A'*r_(k))))
%   t2=(1+sqrt(1+4*t1^2))/2;
%   y_(k+1)=x_(k+1)+(t1-1)/t2*(x_(k+1)-x_(k));
%   t1=t2;
%   r_(k+1)=b-A*y_(k+1)
% end
% *************************************

%% parameters

dim=options.dim;
max_outter_it=options.max_outter_it;
Lambda=options.Lambda;
L=options.L;
t=options.t;
tol=options.tol;
error_upper_bound=options.error_upper_bound;
time_upper_bound=options.time_upper_bound;
display_in_figure=options.display_in_figure;

x=zeros(dim);
y=zeros(dim);
if display_in_figure == true
    error=zeros(1,max_outter_it);
    time=zeros(1,max_outter_it);
end

%% funcs

proxR=func_hand.proxR;
dF=func_hand.dF;

%% iterations

timer=tic;
for i = 1 : max_outter_it
    
    switch options.W_config
        case 'None'
            x_new=proxR((y-2/L*dF(y)),Lambda/L);
    end
    
    t_new=(1+sqrt(1+4*t^2))/2;
    y_new=x_new+(t-1)/t_new*(x_new-x);

    err=val_error(y_new,y,b,options,func_hand);

    if display_in_figure == true
        error(i)=err;
    end

    y=y_new;
    x=x_new;
    t=t_new;

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