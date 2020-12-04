steve = {"Name": "Steve",
         "Homework": [90, 97, 75, 92],
         "Quizzes": [88, 40, 94],
         "Tests": [75, 90]}
alice = {"Name": "Alice",
         "Homework": [100, 92, 98, 100],
         "Quizzes": [88, 40, 94],
         "Tests": [75, 90]}
tyler = {"Name": "Tyler",
         "Homework": [0, 87, 75, 22],
         "Quizzes": [0, 75, 78],
         "Tests": [100, 100]}
print(steve)
print(alice)
print(tyler)


students = []
students = [steve, alice, tyler]

for i in students:
    print(f"Name: {i['Name']}\nHomework: {i['Homework']}\nQuizzes: {i['Quizzes']}\nTests: {i['Tests']}")


numbers = []
def average(numbers):
    return sum(numbers) / len(numbers)

def get_weighted_average(student):
        homework_average = average(student["Homework"])
        quiz_average = average(student["Quizzes"])
        test_average = average(student["Tests"])
        weighted_score = homework_average*.1 + quiz_average*.3 + test_average*.6
        return weighted_score

print(f"Steve's: {get_weighted_average(steve)}")
print(f"Tyler's: {get_weighted_average(tyler)}")
print(f"Alice's: {get_weighted_average(alice)}")

def get_letter_grade(score):
    if score >= 90:
        return 'A'
    elif score >= (80):
        return 'B'
    elif score >= (70):
        return 'C'
    elif score >= (60):
        return 'D'
    else:
        return 'F'

print(get_letter_grade(50))
print(get_letter_grade(100))
print(get_letter_grade(72.5))

for i in students:
    print(f"Name: {i['Name']}'s weighted score is {get_weighted_average(i)}")
    print(f"Name: {i['Name']}'s letter grade is: {get_letter_grade(get_weighted_average(i))}")