A=imread('Ad.tif');
B=imread('Bd.tif');
C=imread('Cd.tif');
D=imread('Dd.tif');
vals=round([mean(A(:)),mean(B(:)),mean(C(:)),mean(D(:))]);
comps=[0.1,0.05,0.1,0.1]
disp(vals);