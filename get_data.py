print "First we'll get a list of every player in the NBA and their respective positions off of a list on yahoo sports:\nhttp://sports.yahoo.com/nba/players?type=position&c=NBA&pos=\n\nThis will be the data we will work from."
raw_input('enter to continue')

import scrapenames
scrapenames.scrape('namelist.txt')

print "\n\nNext, we will get stastics for each of these players off of the NBA website. Some won't work because of differences in what yahoo and NBA.com think the players' names are, but not enough for it to matter. This step might take a while."
raw_input('enter to continue')

import scrapedata
scrapedata.scrape('namelist.txt')

print "\n\nNow we'll prepare the data for use in the classifier, formatting it and separating it into a training and test set, at 70/30 ratio."
raw_input('enter to continue')

import processnames
processnames.process('database')

print "\n\nThe file called 'database' is now ready. For use."
