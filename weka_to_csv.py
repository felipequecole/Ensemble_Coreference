def convert_to_csv():
    entrada = open("saida.csv", "r")
    saida = open("coreference_2.csv", "w")
    instances = 0
    for i in entrada.readlines():
        split = i.split(",")
        cont = 0
        for part in split:
            if (cont > 1):
                if (cont == len(split) -1):
                    saida.write(part)
                else:
                    saida.write(part + ',')
            elif (cont == 1):
                saida.write(str(instances) + ',')
            cont += 1
        instances += 1

def get_names():
    arquivo = open('saida.csv', 'r')
    saida = open('nomes.csv', 'w')
    lines = arquivo.readlines()
    for line in lines:
        linesplit = line.split(",")
        saida.write(linesplit[0] + "," + linesplit[1]+"\n")

if __name__ == '__main__':
    get_names()