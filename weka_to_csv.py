def convert_to_csv():
    entrada = open("saida.csv", "r")
    saida = open("coreference_ready.csv", "w")

    for i in entrada.readlines():
        split = i.split(",")
        cont = 0
        for part in split:
            if (cont > 1):
                if (cont == len(split) -1):
                    saida.write(part)
                else:
                    saida.write(part + ',')
            cont += 1

if __name__ == '__main__':
    convert_to_csv()