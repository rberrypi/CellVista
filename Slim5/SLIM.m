function [phi,L1] = SLIM(A,B,C,D,region,L1,rect,At)
    Gs=D-B; Gc=A-C;
    del_phi=atan2(Gs,Gc);
    %beta
    %L=(Gc+Gs)./(sin(del_phi)+cos(del_phi))/4; %E0*E1 (sqrt(Io.IR))
    L=((Gc.^2 + Gs.^2).^(1/2))/4; %Or...
    g1=(A+C)/2; %E0^2+E1^2=s (Io+IR)
    g2=L.*L; %E0^2*E1^2=p (Io.IR)
    x1=g1/2-sqrt(g1.*g1-4*g2)/2;
    x2=g1/2+sqrt(g1.*g1-4*g2)/2; %solutions (x1= E0^2 and x2= E1^2)
    beta1=sqrt(x1./x2);
    %get constant from  average over pixels
    switch region % Average
        case 1 %probably unused
            croped =@(img) img(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3));      
            cL=croped(L);
            cbeta1=croped(beta1);
            L1=real(nanmean(cbeta1(:)))/nanmean(cL(:)) ; %<(E0/E1)>/<(E0.E1)>=<1/E1^2>)
        case 2 %Whole
            cL=L;
            cbeta1=beta1;
            L1=real(nanmean(cbeta1(:)))/nanmean(cL(:)) ; %<(E0/E1)>/<(E0.E1)>=<1/E1^2>)
           %Pass though value
    end
    %LL=; %<1/E1^2>.E0.E1=E0/E1
    beta=L1*L*At; %real beta, 2.45 for 40X and 4 for 10X for fSLIM
    phi=atan2(beta.*sin(del_phi),1+beta.*cos(del_phi));
    phi(isnan(phi)) = 0 ;
    %phi=single(inpaint_nans(double(phi),3));
end