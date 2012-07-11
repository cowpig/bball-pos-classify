import statget
import traceback
import sys

'''
iterates through a file full of player names and
in the format:
Player Name\tPosition\n
and adds their stats to the file
'''

def scrape(filename):
    with open(filename) as f:
        names = f.read()
        names = names.strip().split('\n')
        
        with open('database', 'w') as d:
            for name in names:
                n = name.split('\t')
                try:
                    x = statget.stats(n[0])
                    d.write(n[1] + '\t' + n[0])
                    for stat in x:
                        d.write('\t' + str(stat))
                    d.write('\n')
                    print "Sucess for: %s" % n
                except:
                    print "Failed getting stats for: %s\n" % n
                    #print traceback.format_exc()
                    #sys.exit()
