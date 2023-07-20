GaussKernel3D = A.*exp(-(((X.^2)/(2.*Sigma.^2)) + ((Y.^2)/(2.*Sigma.^2)) + ((Z.^2)/(2.*Sigma.^2))))

dGauss3Dx = A.*X.*2.*(1/(2.*Sigma.^2)).*exp(-(((X.^2)/(2.*Sigma.^2)) + ((Y.^2)/(2.*Sigma.^2)) + ((Z.^2)/(2.*Sigma.^2))))
dGauss3Dy = A.*Y.*2.*(1/(2.*Sigma.^2)).*exp(-(((X.^2)/(2.*Sigma.^2)) + ((Y.^2)/(2.*Sigma.^2)) + ((Z.^2)/(2.*Sigma.^2))))
dGauss3Dz = A.*Z.*2.*(1/(2.*Sigma.^2)).*exp(-(((X.^2)/(2.*Sigma.^2)) + ((Y.^2)/(2.*Sigma.^2)) + ((Z.^2)/(2.*Sigma.^2))))