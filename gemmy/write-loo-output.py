import os
import sys

dir = sys.argv[1]

thresholds = [1.1]
#jthresholds = [1.1]

#MAX_K_NEIGHBORS_LIST = [10, 20, 30, 40, 50]

#thresholds = [0.05, 0.01]
#jthresholds = [0.05, 0.01]

#MAX_K_NEIGHBORS_LIST = [5, 10]


def getFileClass(fileName):
    metadataFileName = fileName.rsplit('.')[0]+'.metadata'
    metadataFile = open(metadataFileName, 'r')
    fileClass = 'malware'
    for line in metadataFile.readlines():
        if 'clean' in line:
            fileClass = 'clean'

    return fileClass




def main():   

        for root,dirs,files in os.walk(dir):
            for file in [i for i in files if i.endswith('n-2k-ref-20k-f59') and os.path.getsize(os.path.join(root,i))!=0]:


                # Petteri's 1NN Approach
                estimatedNeighborsFile = open(os.path.join(root,file), 'r')
                print file
                line = estimatedNeighborsFile.readline()
                candidate,distance = line.rstrip().split(' ')
                # If we are doing LOO, this might prove useful...
                if file.split('.')[0] == candidate.split('.')[0].split('/')[-1]:
                    line = estimatedNeighborsFile.readline()
                    candidate,distance = line.rstrip().split(' ')
                lines = estimatedNeighborsFile.readlines()
                #lines.reverse()
                distance = float(distance)
                candidatePath = '/share/work/ymiche/F-Secure/reference-set/'+candidate[0:2]+'/'+candidate[2:4]+'/'
                estimatedClass = getFileClass(os.path.join(candidatePath,candidate))     

                closestDifferentNeighbor = "None"
                closestDifferentNeighborClass = "None"
                closestDifferentNeighborDistance = 1.0
                closestDifferentNeighborK = "None"
                for line,k in zip(lines, range(2,lines.__len__()+2)):
                    currentNeighbor, currentDistance = line.rstrip().split(' ')
                    currentNeighborPath='/share/work/ymiche/F-Secure/reference-set/'+currentNeighbor[0:2]+'/'+currentNeighbor[2:4]+'/'
                    currentClass = getFileClass(os.path.join(currentNeighborPath,currentNeighbor))
                    if currentClass != estimatedClass:
                        closestDifferentNeighbor = currentNeighbor.split('.')[0].split('/')[-1]
                        closestDifferentNeighborClass = currentClass
                        closestDifferentNeighborDistance = currentDistance
                        closestDifferentNeighborK = k
                        break

                estimatedNeighborsFile.close()

                fileHash = file.split('.')[0]

                realClass = getFileClass(os.path.join(root, file))
                if realClass == 'malware':
                    realVerdict = '1'
                else:
                    realVerdict = '-1'

                for threshold in thresholds:
                    outputFileName = '/share/work/ymiche/F-Secure/Outputs/loo-output-petteri-threshold-'+str(threshold)+'-50k.out'
                    outputFile = open(outputFileName, 'a')
                    if distance>threshold:
                        verdict = '0'
                    else:
                        if estimatedClass == 'malware':
                            verdict = '1'
                        else:
                            verdict = '-1'
                    
                    outputFile.write(fileHash+' '+realVerdict+' '+verdict+' '+str(distance)+' '+str(closestDifferentNeighbor)+' '+str(closestDifferentNeighborK)+' '+str(closestDifferentNeighborDistance)+'\n')
                    outputFile.close()
                

                # Jozsef's KNN. Here uses as many neighbors as available
                #for MAX_K_NEIGHBORS in MAX_K_NEIGHBORS_LIST:
                #    estimatedNeighborsFile = open(os.path.join(root,file), 'r')
                #    lines = estimatedNeighborsFile.readlines()
                #    lines = lines[:MAX_K_NEIGHBORS+1]
                #    lines.reverse()
                #    candidate,distance = lines[0].rstrip().split(' ')
                #    distance = float(distance)
                #    for jthreshold in jthresholds:
                #        outputFileName = '/share/work/ymiche/F-Secure/Outputs/loo-output-jozsef-k-'+str(MAX_K_NEIGHBORS)+'-threshold-'+str(jthreshold)+'.out'
                #        outputFile = open(outputFileName, 'a')
                #        if distance > jthreshold:
                #            verdict = '0'
                #        else:
                #            lastNNClass = getFileClass(candidate)
                #            flagNoUnanimity = 0
                #            for line in lines[:-2]:
                #                candidate, distance = line.rstrip().split(' ')
                #                if file.split('.')[0] != candidate.split('.')[0].split('/')[-1]:
                #                    if getFileClass(candidate) != lastNNClass:
                #                        flagNoUnanimity = 1
                #            
                #            if flagNoUnanimity == 1:
                #                verdict = '0'
                #            else:    
                #                estimatedClass = lastNNClass        
                #                if estimatedClass == 'malware':
                #                    verdict = '1'
                #                else:
                #                    verdict = '-1'
                #        
                #        outputFile.write(fileHash+' '+verdict+'\n')
                #        outputFile.close()
                #                
             


if __name__ == '__main__':
    main()
