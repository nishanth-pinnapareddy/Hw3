import numpy as np
import sys


def get_random_other_character(char, ):
    if char == 'A':
        return str(np.random.choice(['B', 'C', 'D'], 1)[0])
    elif char == 'B':
        return str(np.random.choice(['A', 'C', 'D'], 1)[0])
    elif char == 'C':
        return str(np.random.choice(['A', 'B', 'D'], 1)[0])
    else:
        return str(np.random.choice(['A', 'B', 'C'], 1)[0])


def mutate(string, rate, ends):
    for i in range(len(string)):
        if (0 <= i < ends) or ((len(string)-ends) <= i < len(string)):
            decision = np.random.choice(['mutate', 'not_mutate'], 1, p=[rate, 1.0-rate])
            if decision == 'mutate':
                string[i] = get_random_other_character(string[i])
        else:
            string[i] = get_random_other_character(string[i])

    return string


def randomly_generate_stick_palindrome():
    letters = ['A', 'B', 'C', 'D']
    first_half = np.random.choice(letters, 20)
    last_half = []

    for letter in first_half:
        if letter == 'A':
            last_half.append('C')
        elif letter == 'B':
            last_half.append('D')
        elif letter == 'C':
            last_half.append('A')
        else:
            last_half.append('B')

    first_half = np.ndarray.tolist(first_half)
    last_half = last_half[::-1]  # w reverse
    return first_half + last_half


# python sticky_snippet_generator.py num_snippets mutation_rate from_ends output_file
if __name__ == "__main__":
    num_snippets = int(sys.argv[1])
    mutation_rate = float(sys.argv[2])
    from_ends = int(sys.argv[3])
    output_file = sys.argv[4]

    initial_stick_palindrome = randomly_generate_stick_palindrome()
    while num_snippets:
        string_after_mutation = mutate(initial_stick_palindrome, mutation_rate, from_ends)
        with open(output_file, 'a') as f:
            f.write(''.join(string_after_mutation) + '\n')
        f.close()
        num_snippets -= 1
