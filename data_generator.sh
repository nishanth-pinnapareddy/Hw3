#!/bin/sh

# Generate training data
i=0
num_snippets=(5000 10000 20000 60000)

while [ $i -lt 4 ]
do
    folder_name=train_folder_0`expr $i + 1`

    # remove existing data
    rm -rf $folder_name

    # train_folder_01
    mkdir $folder_name

    if [ $i -eq 3 ]
    then
        # non-stick
        python sticky_snippet_generator.py ${num_snippets[$i]} 0 0 $folder_name/non_stick_out.txt
        break
    fi

    # non-stick
    python sticky_snippet_generator.py ${num_snippets[$i]} 0 0 $folder_name/non_stick_out.txt

    # 12-stick class
    python sticky_snippet_generator.py `expr ${num_snippets[$i]} / 2` 0 1 $folder_name/12_stick_out.txt
    python sticky_snippet_generator.py `expr ${num_snippets[$i]} / 2` 0 2 $folder_name/12_stick_out.txt

    # 34-stick class
    python sticky_snippet_generator.py `expr ${num_snippets[$i]} / 2` 0 3 $folder_name/34_stick_out.txt
    python sticky_snippet_generator.py `expr ${num_snippets[$i]} / 2` 0 4 $folder_name/34_stick_out.txt

    # 56-stick class
    python sticky_snippet_generator.py `expr ${num_snippets[$i]} / 2` 0 5 $folder_name/56_stick_out.txt
    python sticky_snippet_generator.py `expr ${num_snippets[$i]} / 2` 0 6 $folder_name/56_stick_out.txt

    # 78-stick class
    python sticky_snippet_generator.py `expr ${num_snippets[$i]} / 2` 0 7 $folder_name/78_stick_out.txt
    python sticky_snippet_generator.py `expr ${num_snippets[$i]} / 2` 0 8 $folder_name/78_stick_out.txt

    # stick palindrome
    python sticky_snippet_generator.py ${num_snippets[$i]} 0 20 $folder_name/full_stick_out.txt

    i=`expr $i + 1`
done

# Generate test data
i=0
mutation_rate=(0.2 0.4 0.6)
num_snippets=5000

while [ $i -lt 3 ]
do
    folder_name=test_folder_0`expr $i + 1`

    # remove existing data
    rm -rf $folder_name

    # train_folder_01
    mkdir $folder_name

    # non-stick
    python sticky_snippet_generator.py ${num_snippets} ${mutation_rate} 0 $folder_name/non_stick_out.txt

    # 12-stick class
    python sticky_snippet_generator.py `expr ${num_snippets} / 2` ${mutation_rate} 1 $folder_name/12_stick_out.txt
    python sticky_snippet_generator.py `expr ${num_snippets} / 2` ${mutation_rate} 1 $folder_name/12_stick_out.txt

    # 34-stick class
    python sticky_snippet_generator.py `expr ${num_snippets} / 2` ${mutation_rate} 3 $folder_name/34_stick_out.txt
    python sticky_snippet_generator.py `expr ${num_snippets} / 2` ${mutation_rate} 4 $folder_name/34_stick_out.txt

    # 56-stick class
    python sticky_snippet_generator.py `expr ${num_snippets} / 2` ${mutation_rate} 5 $folder_name/56_stick_out.txt
    python sticky_snippet_generator.py `expr ${num_snippets} / 2` ${mutation_rate} 6 $folder_name/56_stick_out.txt

    # 78-stick class
    python sticky_snippet_generator.py `expr ${num_snippets} / 2` ${mutation_rate} 7 $folder_name/78_stick_out.txt
    python sticky_snippet_generator.py `expr ${num_snippets} / 2` ${mutation_rate} 8 $folder_name/78_stick_out.txt

    # stick palindrome
    python sticky_snippet_generator.py ${num_snippets} ${mutation_rate} 20 $folder_name/full_stick_out.txt

    i=`expr $i + 1`
done