function e=val_error(x_new,x,b,options,func_hand)

if options.x_true_known
    x_true=options.x_true;
end

F=func_hand.F;
R=func_hand.R;
J=func_hand.J;

switch options.stop_criterion
    case 'change_rel_norm'
        e=norml2(x_new-x)/norml2(b);
    case 'rel_norm'
        e=norml2(x_new)/norml2(b);
    case 'rel_norm_from_truth'
        e=norml2(x_new-x_true)/norml2(b);
    case 'change_abs_norm'
        e=norml2(x_new-x);
    case 'abs_norm'
        e=norml2(x_new);
    case 'abs_norm_from_truth'
        e=norml2(x_new-x_true);
    case 'obj_J'
        e=J(x_new);
    case 'obj_F'
        e=F(x_new);
    case 'obj_R'
        e=R(x_new);
    case 'change_obj_J'
        e=J(x_new)-J(x);
    case 'change_obj_F'
        e=F(x_new)-F(x);
    case 'change_obj_R'
        e=R(x_new)-R(x);
    case 'max_outter_it'
        e=inf;
end

end