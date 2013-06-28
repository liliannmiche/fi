import os
import sys

dir = sys.argv[1]
threshold = float(sys.argv[2])
jthreshold = float(sys.argv[3])

MAX_K_NEIGHBORS = 30

def printStats(TotalPositives, TotalNegatives, truePositives, trueNegatives, falseNegatives, falsePositives, unknownPositives, unknownNegatives,\
                        JTotalPositives, JTotalNegatives, JtruePositives, JtrueNegatives, JfalseNegatives, JfalsePositives, JunknownPositives, JunknownNegatives):
    if (truePositives+falseNegatives != 0) and (falsePositives+trueNegatives != 0):
        print '---------------------------------------------------------\n\
################ Petteris Method ################\n\
Gone through {0} Clean and {1} Malware ({2} samples):\n\
In absolute numbers:\n\
\t\tPredicted\n\
\t\tMalware\tClean\tUnknown\n\
Actual\tMalware\t{3}\t{4}\t{5}\t{6}\n\
\tClean\t{7}\t{8}\t{9}\t{10}\n\
\t\t{11}\t{12}\t{13}\n\
In percentages:\n\
\t\tPredicted\n\
\t\tMalware\tClean\n\
Actual\tMalware\t{14}\t{15}\n\
\tClean\t{16}\t{17}\n\
Coverage: {18}\n'.format(TotalNegatives, TotalPositives, TotalPositives+TotalNegatives,\
                            truePositives, falseNegatives, unknownPositives, truePositives+falseNegatives,\
                            falsePositives, trueNegatives, unknownNegatives, falsePositives+trueNegatives,\
                            truePositives+falsePositives, falseNegatives+trueNegatives, unknownNegatives+unknownPositives,\
                            float(100*truePositives)/(truePositives+falseNegatives),\
                            float(100*falseNegatives)/(truePositives+falseNegatives),\
                            float(100*falsePositives)/(falsePositives+trueNegatives), \
                            float(100*trueNegatives)/(falsePositives+trueNegatives),\
                            (100*(1.0-float(unknownPositives+unknownNegatives)/float(TotalPositives+TotalNegatives))))

    if (JtruePositives+JfalseNegatives != 0) and (JfalsePositives+JtrueNegatives != 0):
        print '################ Jozsefs Method ################\n\
Gone through {0} Clean and {1} Malware ({2} samples):\n\
In absolute numbers:\n\
\t\tPredicted\n\
\t\tMalware\tClean\tUnknown\n\
Actual\tMalware\t{3}\t{4}\t{5}\t{6}\n\
\tClean\t{7}\t{8}\t{9}\t{10}\n\
\t\t{11}\t{12}\t{13}\n\
In percentages:\n\
\t\tPredicted\n\
\t\tMalware\tClean\n\
Actual\tMalware\t{14}\t{15}\n\
\tClean\t{16}\t{17}\n\
Coverage: {18}\n'.format(JTotalNegatives, JTotalPositives, JTotalPositives+JTotalNegatives,\
                            JtruePositives, JfalseNegatives, JunknownPositives, JtruePositives+JfalseNegatives,\
                            JfalsePositives, JtrueNegatives, JunknownNegatives, JfalsePositives+JtrueNegatives,\
                            JtruePositives+JfalsePositives, JfalseNegatives+JtrueNegatives, JunknownNegatives+JunknownPositives,\
                            float(100*JtruePositives)/(JtruePositives+JfalseNegatives),\
                            float(100*JfalseNegatives)/(JtruePositives+JfalseNegatives),\
                            float(100*JfalsePositives)/(JfalsePositives+JtrueNegatives), \
                            float(100*JtrueNegatives)/(JfalsePositives+JtrueNegatives),\
                            (100*(1.0-float(JunknownPositives+JunknownNegatives)/float(JTotalPositives+JTotalNegatives))))



def getFileClass(fileName):
    metadataFileName = fileName.rsplit('.')[0]+'.metadata'
    metadataFile = open(metadataFileName, 'r')
    fileClass = 'malware'
    for line in metadataFile.readlines():
        if 'clean' in line:
            fileClass = 'clean'

    return fileClass




def main():
    while True:
        truePositives = 0
        trueNegatives = 0
        falsePositives = 0
        falseNegatives = 0  

        TotalNegatives = 0
        TotalPositives = 0  

        unknownNegatives = 0
        unknownPositives = 0    

        JtruePositives = 0
        JtrueNegatives = 0
        JfalsePositives = 0
        JfalseNegatives = 0

        JunknownNegatives = 0
        JunknownPositives = 0   

        for root,dirs,files in os.walk(dir):
            for file in [i for i in files if i.endswith('neighbors')]:
                # Petteri's 1NN Approach
                estimatedNeighborsFile = open(os.path.join(root,file), 'r')
                line = estimatedNeighborsFile.readline()
                candidate,distance = line.rstrip().split(' ')
                # If we are doing LOO, this might prove useful...
                if file.split('.')[0] == candidate.split('.')[0].split('/')[-1]:
                    line = estimatedNeighborsFile.readline()
                    candidate,distance = line.rstrip().split(' ')
                estimatedNeighborsFile.close()
                distance = float(distance)
                estimatedClass = getFileClass(candidate)     

                realClass = getFileClass(os.path.join(root, file))

                if realClass == 'malware':
                    TotalPositives += 1
                    if distance > threshold:
                        unknownPositives += 1
                    else:
                        if estimatedClass == 'malware':
                            truePositives += 1
                        elif estimatedClass == 'clean':
                            falseNegatives += 1
                        else:
                            print 'Class unknown: '+str(estimatedClass)     

                elif realClass == 'clean':
                    TotalNegatives += 1
                    if distance > threshold:
                        unknownNegatives += 1
                    else:
                        if estimatedClass == 'malware':
                            falsePositives += 1
                        elif estimatedClass == 'clean':
                            trueNegatives += 1
                        else:
                            print 'Class unknown: '+str(estimatedClass)     

                else:
                    print 'Class unknown: '+str(realClass)
                    break
                    exit
                

                # Jozsef's KNN. Here uses as many neighbors as available
                estimatedNeighborsFile = open(os.path.join(root,file), 'r')
                lines = estimatedNeighborsFile.readlines()
                lines = lines[:MAX_K_NEIGHBORS+1]
                lines.reverse()
                candidate,distance = lines[0].rstrip().split(' ')
                distance = float(distance)
                if distance > jthreshold:
                    if realClass == 'malware':
                        JunknownPositives += 1
                    else:
                        JunknownNegatives += 1
                else:
                    lastNNClass = getFileClass(candidate)
                    flagNoUnanimity = 0
                    for line in lines[:-2]:
                        candidate, distance = line.rstrip().split(' ')
                        if file.split('.')[0] != candidate.split('.')[0].split('/')[-1]:
                            if getFileClass(candidate) != lastNNClass:
                                flagNoUnanimity = 1
                    
                    if flagNoUnanimity == 1:
                        if realClass == 'malware':
                            JunknownPositives += 1
                        else:
                            JunknownNegatives += 1
                    else:    
                        estimatedClass = lastNNClass   

                        if realClass == 'malware':
                            if estimatedClass == 'malware':
                                JtruePositives += 1
                            elif estimatedClass == 'clean':
                                JfalseNegatives += 1
                            else:
                                print 'Class unknown: '+str(estimatedClass)         

                        elif realClass == 'clean':
                            if estimatedClass == 'malware':
                                JfalsePositives += 1
                            elif estimatedClass == 'clean':
                                JtrueNegatives += 1
                            else:
                                print 'Class unknown: '+str(estimatedClass)         

                        else:
                            print 'Class unknown: '+str(realClass)
                            break
                            exit
                estimatedNeighborsFile.close()
                
                if TotalPositives != 0 or TotalNegatives != 0:
                    printStats(TotalPositives, TotalNegatives, truePositives, trueNegatives, falseNegatives, falsePositives, unknownPositives, unknownNegatives,\
                                TotalPositives, TotalNegatives, JtruePositives, JtrueNegatives, JfalseNegatives, JfalsePositives, JunknownPositives, JunknownNegatives)



if __name__ == '__main__':
    main()