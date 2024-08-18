function [x,error,time,i] = LBA(b,options,func_hand)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LBA 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x_0=0,r_0=b
% for k
%   x_(k+1)=alpha*W^(-1)*(T_(Lambda)(W*(v_(k))))
%   v_(k+1)=v_(k)-AT*(A*x_(k+1)-b)
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% parameters

dim=options.dim;
max_outter_it=options.max_outter_it;
Lambda=options.Lambda;
alpha=options.alpha;
tol=options.tol;
error_upper_bound=options.error_upper_bound;
time_upper_bound=options.time_upper_bound;
display_in_figure=options.display_in_figure;

x=zeros(dim);
v=zeros(dim);
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
            x_new=alpha*proxR((v),Lambda);
    end

    v=v-dF(x_new);

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