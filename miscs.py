import csv

def convertCSV():
    with open('submission.csv', 'r') as f:
        txt = f.read()
        lines = txt.split('\n')
        f.close()

    with open('sub.csv', 'w') as f:
        output = []
        for line in lines[1:-1]:
            line = line.split(',')
            locs = line[4].split('|')
            x, w = locs[0].split(' ')
            y, h = locs[1].split(' ')
            x = int(float(x))
            w = int(float(w))
            y = int(float(y))
            h = int(float(h))
            s = '{},{},{},{},'.format(line[0], line[1], line[2], line[3]) + '{} {}|{} {}'.format(x,w,y,h)
            output.append(s)
        out = '\n'.join(output)
        f.write(out)
        f.close()

convertCSV()
