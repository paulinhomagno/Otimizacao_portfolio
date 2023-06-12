# função para truncar as casa decimais
def truncar(num,n):
    numeros = str(num)
    
    for x in range(len(numeros)):
        if numeros[x] == '.':
            try:
                return float(numeros[:x+n+1])
            except:
                return float(numeros)      
            
    return float(numeros)