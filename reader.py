# structure data[reviewer] = [(review, scale), ...]
# if path is not predefined, then search root file
def readData(path = "./scaledata", scale = 3):
    data = {}
    reviewers = ["Dennis+Schwartz", "James+Berardinelli", "Scott+Renshaw", "Steve+Rhodes"]
    for reviewer in reviewers:
        
        scales = []
        scaleFileName = "label.{}class.".format(scale) + reviewer
        with open(path + '/' + reviewer + '/' + scaleFileName) as f:
            for line in f:
                scales.append(int(line))
        
        reviews= []
        reviewFileName = "subj." + reviewer
        with open(path + '/' + reviewer + '/' + reviewFileName) as f:
            for line in f:
                reviews.append(line)
        
        data[reviewer] = list(zip(reviews, scales))

    return data


        

if __name__ == '__main__':
    data = readData()
    # for reviewer in data:
    #     print(data[reviewer][0][0])  # review of the 1st entry
    #     print(data[reviewer][0][1])  # scale of the 1st entry
        

        # all reviews from a reviewer
        # reviews = [entry[0] for entry in data[reviewer]]
        # print(reviews)

        # all scales from a reviewer
        # scales = [entry[1] for entry in data[reviewer]]
        # print(scales)