# Import python modules
import numpy 
import kaggle
import matplotlib.pyplot as plt
from decimal import Decimal


def round_vector(v):
    data = [float(Decimal("%.4f" % e)) for e in v]
    return data

def p_JgivenAlpha(J,alpha):
    if J[0] == 1:
        return 0
    else:
        prev = 0
        prod = 1
        for j in J[1:]:
            if j==prev:
                prod *= alpha
            else:
                prev = j
                prod *= (1-alpha)
    return prod

def p_BgivenJ(B,J):
    result = 1.0
    #assume len(B)==len(J)
    for i in range(len(B)):
        if J[i] == 0:
            if B[i] == 0:
                result *= 0.2
            else:
                result *= 0.8
        else:
            if B[i] == 0:
                result *= 0.9
            else:
                result *= 0.1
    return result

def p_alpha(alpha):
    if 0<=alpha and alpha<=1:
        return 1
    else:
        return 0
    
def p_joint(a,J,B):
    return p_alpha(a)*p_BgivenJ(B,J)*p_JgivenAlpha(J,a)

def propose_J(J):
    J_new = []
    index = numpy.random.randint(0,len(J))
    for i in range(len(J)):
        if i == index:
            J_new.append((J[i]+1)%2)
        else:
            J_new.append(J[i])
    return J_new

def drawJ(B,a,iteration):
    J = numpy.zeros(len(B))
    J_mean = J
    for i in range(iteration):
        J_prime = propose_J(J)
        acceptance_ratio = p_joint(a,J_prime,B)/p_joint(a,J,B)
        if numpy.random.rand() < acceptance_ratio:
            J = J_prime
        J_mean = J_mean + J
    return J_mean/iteration

def propose_alpha(alpha):
    return numpy.random.rand()

def drawAlpha(J,B,iteration,graph=False):
    a = 0
    a_mean = a
    a_vector = []
    for i in range(iteration):
        a_prime = propose_alpha(a)
        d = p_joint(a,J,B)
        if d==0:
            d=1e-100
        acceptance_ratio = p_joint(a_prime,J,B)/d
        if numpy.random.rand() < acceptance_ratio:
            a = a_prime
        a_vector.append(a)
        a_mean = a_mean + a
    if graph==True:
        bins = numpy.arange(0,1,.1)
        plt.hist(a_vector,bins=bins)
        plt.ylabel("freq alpha") 
        plt.xlabel("iterations") 
        plt.xlim(0,1) 
        plt.ylim(0,5000) 
        plt.title("alpha freq")  
        plt.savefig("../Figures/1h.png")
        plt.show()
        
    return a_mean/iteration

def proposal(a,J):
    return propose_alpha(a), propose_J(J)
def MCMC(B,iteration):
    a = 0
    a_mean = a
    J = numpy.zeros(len(B))
    J_mean = J
    a_vector=[]
    for i in range(iteration):
        a_prime,J_prime = proposal(a,J)
        d = p_joint(a,J,B)
        if d==0:
            d=1e-100
        acceptance_ratio = p_joint(a_prime,J_prime,B)/d
        if numpy.random.rand() <= acceptance_ratio:
            a = a_prime
            J = J_prime
        a_mean += a
        J_mean += J
        a_vector.append(a)
    return J_mean/iteration, a_mean/iteration, a_vector
def p_next_black(jar,a):
    return p_alpha(a)*a*p_BgivenJ([1],[jar]) + p_alpha(1-a)*(1-a)*p_BgivenJ([1],[(jar+1)%2])

def MCMC_predict(B,iterations):
    a = 0
    J = numpy.zeros(len(B))
    prob = 0
    for i in range(iterations):
        a_prime,J_prime = proposal(a,J)
        d = p_joint(a,J,B)
        if d==0:
            d=1e-100
        acceptance_ratio = p_joint(a_prime,J_prime,B)/d
        if numpy.random.rand() <= acceptance_ratio:
            a = a_prime
            J = J_prime
        prob+=p_next_black(J[len(J)-1],a)
    return prob/iterations
################################################
if __name__ == '__main__':
    print('1a through 1l computation goes here ...')
    
    print('1a')
    print('Case 1: ',p_JgivenAlpha([0,1,1,0,1],0.75))
    print('Case 2: ',p_JgivenAlpha([0,0,1,0,1],0.2))
    print('Case 3: ',p_JgivenAlpha([1,1,0,1,0,1],0.2))
    print('Case 4: ',p_JgivenAlpha([0,1,0,1,0,0],0.2))
    
    print('1b')
    print('Case 1: ',p_BgivenJ([1,0,0,1,1],[0,1,1,0,1]))
    print('Case 2: ',p_BgivenJ([0,0,1,0,1],[0,1,0,0,1]))
    print('Case 3: ',p_BgivenJ([1,0,1,1,1,0],[0,1,1,0,0,1]))
    print('Case 4: ',p_BgivenJ([0,1,1,0,1,1],[1,1,0,0,1,1]))
    
    print('1d')
    print('Case 1: ',p_joint(0.75,[0,1,1,0,1],[1,0,0,1,1]))
    print('Case 2: ',p_joint(0.3,[0,1,0,0,1],[0,0,1,0,1]))
    print('Case 3: ',p_joint(0.63,[0,0,0,0,0,1],[0,1,1,1,0,1]))
    print('Case 4: ',p_joint(0.23,[0,0,1,0,0,1,1],[1,1,0,0,1,1,1]))
    
    print('1f')
    print('Case 1: P(J_2=1|alpha,B) =',drawJ([1,0,0,1,1],0.5,10000)[2])
    inds = numpy.arange(2)
    labels = ["jar_0","jar_1"]
    plt.figure(1, figsize=(4,4))  
    plt.bar(inds, [drawJ([1,0,0,1,1],0.5,10000)[2],1-drawJ([1,0,0,1,1],0.5,10000)[2]], align='center') 
    plt.ylabel("P( J_2 | alpha,B )") 
    plt.xlabel("J_2") 
    plt.xlim(-0.5,2) 
    plt.ylim(0,1) 
    plt.title("P( J_2 | alpha,B )")  
    plt.gca().set_xticks(inds)
    plt.gca().set_xticklabels(labels)
    plt.savefig("../Figures/1f.png")
    plt.show()
    print('Case 2: ',round_vector(drawJ([1,0,0,0,1,0,1,1],0.11,10000)))
    print('Case 3: ',round_vector(drawJ([1,0,0,1,1,0,0],0.75,10000)))
    
    print('1h')
    print('Case 1: ',drawAlpha([0,1,0,1,0],[1,0,1,0,1],10000))
    print('Case 2: ',drawAlpha([0,0,0,0,0],[1,1,1,1,1],10000))
    print('Case 3: ',drawAlpha([0,1,1,0,1],[1,0,0,1,1],10000,graph=True))
    print('Case 4: ',float(Decimal("%.4f" % drawAlpha([0,1,1,1,1,1,1,0],[1,0,0,1,1,0,0,1],10000))))
    print('Case 5: ',float(Decimal("%.4f" % drawAlpha([0,1,1,0,1,0],[1,0,0,1,1,1],10000))))
    
    print('1j')
    J,a,a_vector=MCMC([1,1,0,1,1,0,0,0],10000)
    inds = numpy.arange(8)
    labels = ["J_1","J_2","J_3","J_4","J_5","J_6","J_7","J_8"]
    plt.figure(1, figsize=(8,4))  
    plt.bar(inds, J) 
    plt.ylabel("P( J | B )") 
    plt.xlabel("Jar") 
    plt.xlim(0,9) 
    plt.ylim(0,1) 
    plt.title("P( J | B )")  
    plt.gca().set_xticks(inds)
    plt.gca().set_xticklabels(labels)
    plt.savefig("../Figures/1j 21.pdf")
    plt.show()
    
    bins = numpy.arange(0,1,.1)
    plt.hist(a_vector,bins=bins)#,histtype = 'step',bins=bins)#, align='center') 
    plt.ylabel("freq alpha") 
    plt.xlabel("iterations") 
    plt.xlim(0,1) 
    plt.ylim(0,5000) 
    plt.title("alpha freq")  
    plt.savefig("../Figures/1j 22.png")
    plt.show()
    
    plt.figure(figsize=(10,4))
    plt.plot(range(10000),a_vector)
    plt.ylabel("alpha") 
    plt.xlabel("iterations")
    plt.title("alpha vs iterations")  
    plt.xlim(0,10000) 
    plt.ylim(0,1) 
    plt.savefig("../Figures/1j 23.png")
    plt.show()
    
   
    print('1k')
    print('Case 1: ',p_next_black(1,0.6))
    print('Case 2: ',p_next_black(0,0.99))
    print('Case 3: ',p_next_black(0,0.33456))
    print('Case 4: ',p_next_black(0,0.5019))
    
    print('1l')
    print('Test 1: ',MCMC_predict([0,0,1],10000))
    print('Test 2: ',MCMC_predict([0,1,0,1,0,1],10000))
    print('Test 3: ',MCMC_predict([0,1,0,0,0,0,0],10000))
    print('Test 4: ',MCMC_predict([1,1,1,1,1],10000))
	###############################################
    print('1m')
    lengths = [10, 15, 20, 25]
    prediction_prob = list()
    for l in lengths:
        B_array = numpy.loadtxt('../../Data/B_sequences_%s.txt' % (l), delimiter=',', dtype=float)
#        for b in numpy.arange(B_array.shape[0]):
#             prediction_prob.append(MCMC_predict(b,10000))
#             print('Prob of next entry in ', B_array[b, :], 'is black is', prediction_prob[-1])
        for B in B_array:
             prediction_prob.append(MCMC_predict(B,1000000))
             #print('Prob of next entry in ', B_array[b, :], 'is black is', prediction_prob[-1])

	# Output file location
    file_name = '../Predictions/best.csv'

	# Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(numpy.array(prediction_prob), file_name)
#	
#
