function mat = randEig(dim, rangeEig, nComplex)
% Makes a random real matrix with complex eigenvalues. From:
%   https://stackoverflow.com/questions/49087901/generate-random-matrix-with-eigenvalues


if 2*nComplex > dim
    error('Cannot happen');
end

if nComplex
    cMat=diff(rangeEig).*rand(2*nComplex,1)+rangeEig(1);
    cPart=cMat(1:nComplex)*i;
    cMat(1:nComplex)=[];
    cPart=upsample(cPart,2);
    cPart=cPart+circshift(-cPart,1);
    cMat=upsample(cMat,2);
    cMat=cMat+circshift(cMat,1);
    cMat=cMat+cPart;
    cMat=[diff(rangeEig).*rand(dim-2*nComplex,1)+rangeEig(1); cMat];
else
    cMat=diff(rangeEig).*rand(dim,1)+rangeEig(1);
end
D=cMat;
realDform = comp2rdf(diag(D));
P=rand(dim);
mat=P*realDform/P;
end


function dd = comp2rdf(d)
i = find(imag(diag(d))');
index = i(1:2:length(i));
if isempty(index)
    dd=d;
else
    if (max(index)==size(d,1)) | any(conj(d(index,index))~=d(index+1,index+1))
        error(message('Complex conjugacy not satisfied'));
    end
    j = sqrt(-1);
    t = eye(length(d));
    twobytwo = [1 1;j -j];
    for i=index
        t(i:i+1,i:i+1) = twobytwo;
    end
    dd=t*d/t;
end
end