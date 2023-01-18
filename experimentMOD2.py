'EDWIN MAKOVEEV ROUTSKAIA · 3D1'
import sys; import math; import numpy as np
from perceptron import perceptron; from confus import confus; from linmach import linmach

if len(sys.argv) != 4:
    print ('Usage: %s <data> <alphas> <bs>' % sys.argv[0])
    sys.exit(1)

data = np.loadtxt(sys.argv[1])
alphas = np.fromstring(sys.argv[2], sep = ' ')
bs = np.fromstring (sys.argv[3], sep = ' ')

N,L=data.shape 
D=L-1
labs=np.unique(data[:,L-1])
C=labs.size
np.random.seed(23)
perm=np.random.permutation(N)
data=data[perm]
NTr=int(round(.8*N))
train=data[:NTr,:]
M=N-NTr
test=data[NTr:,:]

print ('#      a        b   E   k Ete Ete (%)    Ite (%)')
print ('#------- -------- --- --- --- ------- ----------')

for a in alphas:
    for b in bs:
        w, E, k = perceptron (train, b, a, 1000)
        rl = np.zeros ((M, 1))

        for n in range(M):
            aux = np.concatenate(([1],test[n,:D]))
            rl[n]=labs[linmach(w,aux)]
        'Comprueba si esta clasificada correctamente cada muestra'
        nerr,m=confus(test[:,L-1].reshape(M,1),rl)

        'Numero de errores / muestras de los tests'
        perr = nerr/M
        aux = perr
        perr *= 100

        'Usando un intervalo de confianza del 95%'
        raiz = math.sqrt(aux * (1-aux)/M)
        res = 1.96*raiz
        izq = (aux - res)*100
        der = (aux + res)*100

        izq = round (izq, 1)
        der = round (der, 1)

        imprimir = '['+ str(izq) + ', ' + str(der) + ']'

        print('%8.1f %8.1f %3d %3d %3d %7.1f %10s' % (a,b,E,k,nerr, perr, imprimir))

