import numpy as np


DNA_SIZE = 24           #DNA长度（用多少的二进制数切片）
POP_SIZE = 200          #种群个数
CROSSOVER_RATE = 0.8    #交叉概率
MUTATION_RATE = 0.005   #变异概率
N_GENERATIONS = 50      #迭代次数
# U_BOUND=[-1,1]           #均值的取值范围
# Std_BOUND=[0,2]         #标准差取值范围



# @Author : 徐祖齐
# Description: 
# @brief : 正态函数
# @param : u：均值，std：标准差，x：对应横坐标值
# @ability : 求正态函数在一点的对应值
def f(u,std,x):
    return (1/np.sqrt(2*np.pi*std)) * np.exp(-(x-u)**2/(2*std**2))

 
# @Author : 徐祖齐
# Description: 
# @brief : 积分函数，用来积分正态函数
# @param : f: function, X_LowerBound: 积分下限， X_UpperBound：积分上限，u：均值，std：标准差
# @ability : 
def S_Integral(f,X_LowerBound,X_UpperBound,u=0,std=1,dx = 1e-4):
    S=0
    Time=(X_UpperBound - X_LowerBound)/dx  # 次数
    for i in range(0,int(Time)):
        S+=(f(u,std,i*dx + X_LowerBound) + f(u,std,(i-1)*dx + X_LowerBound))*dx/2
    return S

# @Author : 徐祖齐
# Description: 
# @brief : 最小二乘法算损失函数
# @param : S_Integral：正态函数面积， T_Value：真实面积
# @ability : 
def Loss_Function(S_Integral,T_Value,f,X_LowerBound,X_UpperBound,u=0,std=1):
    return (S_Integral(f,X_LowerBound,X_UpperBound,u,std)-T_Value)**2

def Loss_AllIntegral(Loss_Function,T_Value,X_LowerBound,X_UpperBound,u,std):
    s=0
    for i in range(0,np.size(X_LowerBound)):
        s+=Loss_Function(S_Integral,T_Value[i],f,X_LowerBound[i],X_UpperBound[i],u,std)
    return s

# 计算使损失函数最小的值
def get_minfitness(pop,T_Value,f,X_LowerBound,X_UpperBound,U_BOUND=[-1,1],Std_BOUND=[0,2]): 
    u,std = translateDNA(pop,U_BOUND,Std_BOUND)
    pred = Loss_AllIntegral(Loss_Function,T_Value,X_LowerBound,X_UpperBound,u,std)
    return -(pred - np.max(pred)) + 1e-3

def translateDNA(pop,U_BOUND=[-1,1],Std_BOUND=[0,2]): #pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
	x_pop = pop[:,1::2]#奇数列表示X
	y_pop = pop[:,::2] #偶数列表示y
	
	#pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
	u = x_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(U_BOUND[1]-U_BOUND[0])+U_BOUND[0]
	std = y_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(Std_BOUND[1]-Std_BOUND[0])+Std_BOUND[0]
	return u,std


def crossover_and_mutation(pop, CROSSOVER_RATE = 0.8):
	new_pop = []
	for father in pop:		#遍历种群中的每一个个体，将该个体作为父亲
		child = father		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
		if np.random.rand() < CROSSOVER_RATE:			#产生子代时不是必然发生交叉，而是以一定的概率发生交叉
			mother = pop[np.random.randint(POP_SIZE)]	#再种群中选择另一个个体，并将该个体作为母亲
			cross_points = np.random.randint(low=0, high=DNA_SIZE*2)	#随机产生交叉的点
			child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
		mutation(child)	#每个后代有一定的机率发生变异
		new_pop.append(child)

	return new_pop

def mutation(child, MUTATION_RATE=0.003):
	if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
		mutate_point = np.random.randint(0, DNA_SIZE*2)	#随机产生一个实数，代表要变异基因的位置
		child[mutate_point] = child[mutate_point]^1 	#将变异点的二进制为反转

def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness)/(fitness.sum()) )
    return pop[idx]


def print_info(pop,T_Value,X_LowerBound,X_UpperBound,U_BOUND=[-1,1],Std_BOUND=[0,2]):
	fitness = get_minfitness(pop,T_Value,f,X_LowerBound,X_UpperBound,U_BOUND,Std_BOUND)
	min_fitness_index = np.argmax(fitness)
	print("min_fitness:", fitness[min_fitness_index])
	u,std = translateDNA(pop,U_BOUND,Std_BOUND)
	print("最优的基因型：", pop[min_fitness_index])
	print("(u, std):", (u[min_fitness_index], std[min_fitness_index]))
      
def Get_u_std(pop,T_Value,X_LowerBound,X_UpperBound,U_BOUND=[-1,1],Std_BOUND=[0,2]):
    fitness = get_minfitness(pop,T_Value,f,X_LowerBound,X_UpperBound,U_BOUND,Std_BOUND)
    min_fitness_index = np.argmax(fitness)
    u,std = translateDNA(pop,U_BOUND,Std_BOUND)
    return u[min_fitness_index], std[min_fitness_index]

def GA_NormFitting(Possibilities,X_LowerBound,X_UpperBound,U_BOUND=[-1,1],Std_BOUND=[0,2]):
    T_Value = Possibilities * (X_UpperBound-X_LowerBound)
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*2)) #matrix (POP_SIZE, DNA_SIZE)
    for _ in range(N_GENERATIONS):#迭代N代
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        fitness_min = get_minfitness(pop,T_Value,f,X_LowerBound,X_UpperBound,U_BOUND,Std_BOUND)
        pop = select(pop, fitness_min) #选择生成新的种群
    print_info(pop,T_Value,X_LowerBound,X_UpperBound,U_BOUND,Std_BOUND)
    u,std = Get_u_std(pop,T_Value,X_LowerBound,X_UpperBound,U_BOUND,Std_BOUND)
    return u,std
    