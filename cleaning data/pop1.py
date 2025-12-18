# chatgpt made htis with this prompt make a script that combines moral_action and immoral_action we only want action and each line is either one when we want only one ID,norm,situation,intention,moral_action,moral_consequence,label,immoral_action,immoral_consequence,explanation 3K4J6M3CXFR2F6AYF13KHP3W45NAGD1,It's rude to ditch a date for someone else.,Joan is on a first date with Mitch when she gets a text from her ex-boyfriend who she still loves asking to meet up.,Joan wants to have a fun night.,Joan ignores the text and focuses on enjoying her night with Mitch.,Mitch has a great time on his date with Joan and appreciates her attention.,1,,,Joan is on a first date with Mitch when she gets a text from her ex-boyfriend who she still loves asking to meet up. Joan ignores the text and focuses on enjoying her night with Mitch. 3K4J6M3CXFR2F6AYF13KHP3W45NAGD0,It's rude to ditch a date for someone else.,Joan is on a first date with Mitch when she gets a text from her ex-boyfriend who she still loves asking to meet up.,Joan wants to have a fun night.,,,0,Joan makes up an excuse to end the date early to go see her ex-boyfriend.,Mitch sees her out with her ex later that night and feels hurt and betrayed.,Joan is on a first date with Mitch when she gets a text from her ex-boyfriend who she still loves. She makes up an excuse to end the date early to go see her ex. Mitch sees her out with her ex later that night and feels hurt and betrayed. 3OONKJ5DKDHVQTWO9YGOH9W4D93BO01,It's wrong to abuse a privilege someone gave you.,Justin's mother allows him to stay up as late as he wants as long as he maintains good grades.,Justin wants to have freedom over his schedule.,Justin finds himself naturally wanting to sleep by 10 before busy school days.,Juston maintains good grades and cant occasionally stay up late if a special situation happens.,1,,,Justin's mother allows him to stay up as late as he wants as long as he maintains good grades. Justin finds himself naturally wanting to sleep by 10 before busy school days. 3OONKJ5DKDHVQTWO9YGOH9W4D93BO00,It's wrong to abuse a privilege someone gave you.,Justin's mother allows him to stay up as late as he wants as long as he maintains good grades.,Justin wants to have freedom over his schedule.,,,0,Justin plays video games until 3 am on most days and fails some courses.,Justin loses the privilege and most now go to bed at 9 on weekdays.,Justin's mother allows him to stay up as late as he wants as long as he maintains good grades. Justin plays video games until 3 am on most days and fails some courses. Justin loses the privilege and most now go to bed at 9 on weekdays. 38SKSKU7R2VX7G7X2GSDBFWNDUEILV1,You should always spend time with your parents.,"Margaret's mother is living in a nursing home an hour from her town, and they haven't seen each other in a year.",Margaret wants to talk to her mom.,Margaret makes the hour drive to surprise her mom with a visit.,Margaret's mother is in good spirits to see her daughter and feels more energetic.,1,,,Margaret's mother is living in a nursing home an hour from her town. They haven't seen each other in a year. Margaret makes the hour drive to surprise her mom with a visit. 38SKSKU7R2VX7G7X2GSDBFWNDUEILV0,You should always spend time with your parents.,"Margaret's mother is living in a nursing home an hour from her town, and they haven't seen each other in a year.",Margaret wants to talk to her mom.,,,0,"Margaret calls her mom on the phone and promises to visit soon, but never does.","The next time Margaret calls the nursing home to speak to her mother, they inform her that her mother has passed in the night.","Margaret's mother is living in a nursing home an hour from her town. Margaret calls her mom on the phone and promises to visit soon, but never does. The next time Margaret calls the nursing home, they inform her that her mother has passed in the night." 3G0WWMR1UWI0CEY1R1D1X9RY9Y3NQB0,You should not copy others.,David is taking an exam in his calculus class that he didn't study for.,David wants to pass the exam.,,,0,David copies exam answers from the student sitting next to him.,David's teacher catches him cheating and ends up failing David on the exam.,David is taking an exam in his calculus class that he didn't study for. David copies exam answers from the student sitting next to him. David's teacher catches him cheating and ends up failing David on the exam. 3VE8AYVF8NVIIILO7JDO3DXPD2Z8FK1,It's unfair to bully people,Chad is eating lunch in the school cafeteria and sees that his classmate Billy has several delicious looking cookies on his plate.,Chad wants to taste the cookies.,Chad asks Billy if he can try one of the cookies.,Billy is happy to share and gives Chad a cookie which he enjoys very much.,1,,,Chad is eating lunch in the school cafeteria and sees that his classmate Billy has several delicious looking cookies on his plate. Chad asks Billy if he can try one of the cookies. Billy is happy to share and gives Chad a cookie. 3VE8AYVF8NVIIILO7JDO3DXPD2Z8FK0,It's unfair to bully people,Chad is eating lunch in the school cafeteria and sees that his classmate Billy has several delicious looking cookies on his plate.,Chad wants to taste the cookies.,,,0,Chad pins Billy against the table until he gives him all the cookies.,Chad is sent to the principal's office and later suspended for bullying.,Chad pins Billy against the table until he gives him all the cookies. Chad is sent to the principal's office and later suspended for bullying. 3MX2NQ3YCASGVMT79FC7X0N5FHY5XA1,You shouldn't be a creep toward women.,Gary is walking to attend his class and sees a group of young women on the way.,Gary wants to walk past the group of the women.,Gary says hi and simply walks past the women.,The women also say hi and Gary feels a sense of glee.,1,,,Gary sees a group of young women on the way to class. Gary says hi and simply walks past the women. The women also say hi and Gary feels a sense of glee.

import csv

input_file = "cleaned_data.csv"
output_file = "output.csv"

# Columns in original data
fields_in = [
    "ID","norm","situation","intention",
    "moral_action","moral_consequence","label",
    "immoral_action","immoral_consequence","explanation"
]

# Columns in desired output
fields_out = [
    "ID","norm","situation","intention",
    "action","consequence","label","explanation"
]

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile, fieldnames=fields_in)
    writer = csv.DictWriter(outfile, fieldnames=fields_out)
    writer.writeheader()

    for row in reader:
        # Decide whether to use moral or immoral action
        if row["moral_action"].strip():
            action = row["moral_action"]
            consequence = row["moral_consequence"]
        else:
            action = row["immoral_action"]
            consequence = row["immoral_consequence"]

        outrow = {
            "ID": row["ID"],
            "norm": row["norm"],
            "situation": row["situation"],
            "intention": row["intention"],
            "action": action,
            "consequence": consequence,
            "label": row["label"],
            "explanation": row["explanation"]
        }

        writer.writerow(outrow)

print("Done. Output written to", output_file)
