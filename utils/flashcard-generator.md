Prompt:


[n-flashcards] = 40
[files] = ["machine_learning/P1-ml-oreilly.md", 
           "machine_learning/scikit-learn-cheat-sheet/README.md" 
            ]


I need to study for an interview the contents of the following files: 

[files]

For that generate anki flashards in a csv format with the following columns:

```
question;answer;tags
"What is the definition of independent events in probability?";"Two events A and B are independent if and only if P(A|B) = P(A). For independent events, P(A,B) = P(A)P(B).";probability basic
```

I need to generate at least [n-flashcards] flashcards based on the contents of the files. Please ensure that the questions are clear and concise, and the answers are accurate and informative. 
Create them in an increasing level of difficulty, starting with basic concepts and moving towards more complex topics. The tags should be relevant to the topic of each flashcard. Think about what a hard interview would be like and generate questions that would be asked in such a scenario.

The tags should be relevant to the topic of each flashcard.

Generate under the same directory of the files, a new directory and file called `flashcards\<content>.csv` with the generated flashcards in the specified format. If there's already a file there, expand the existing file with new flashcards, ensuring no duplicates are created.

