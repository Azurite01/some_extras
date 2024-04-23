import numpy as np
import pygame
import sys
import tensorflow as tf

model = tf.keras.models.load_model("digit_recog_model.h5")

pygame.init()

size = width, height = 900, 450
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Digit Identification")

font = "helvetica"
extrasmallFont = pygame.font.SysFont(font, 16)
smallFont = pygame.font.SysFont(font, 20)
largeFont = pygame.font.SysFont(font, 40)
extralargeFont = pygame.font.SysFont(font, 100)

ROWS, COLS = 28, 28
black = (0, 0, 0)
white = (255, 255, 255)


CELL_SIZE = 15

grid = [[0] * COLS for _ in range(ROWS)]
classified = None
numbers = list(range(10))
probabilities = [0] * 10


def draw_table(screen, numbers, probabilities):
    # Set up some constants for the table layout
    x_start = 10

    cell_width = 40
    cell_height = 30

    y_start = height - 82
    spacing = 5

    # Set up colors
    title_surface = smallFont.render("Probailities (in %)", True, white)
    screen.blit(title_surface, (x_start + (cell_width * 10) / 2 - 40, y_start - 30))

    # Loop through each number and probability to draw the table
    for i in range(len(numbers)):
        # Calculate the position for each cell
        x = x_start + i * (cell_width + spacing)
        y_number = y_start
        y_probability = y_start + cell_height + spacing

        # Center the text in each cell
        number_text = smallFont.render(str(numbers[i]), True, white)
        number_text_rect = number_text.get_rect(
            center=(x + cell_width // 2, y_number + cell_height // 2)
        )
        screen.blit(number_text, number_text_rect)

        probability_text = extrasmallFont.render(str(probabilities[i]), True, white)
        probability_text_rect = probability_text.get_rect(
            center=(x + cell_width // 2, y_probability + cell_height // 2)
        )
        screen.blit(probability_text, probability_text_rect)

        # Draw vertical lines
        pygame.draw.line(
            screen,
            white,
            (x + cell_width, y_start),
            (x + cell_width, y_start + cell_height * 2 + spacing),
            2,
        )

    # Draw horizontal lines
    pygame.draw.line(
        screen,
        white,
        (x_start, y_start),
        (x_start + len(numbers) * (cell_width + spacing) - spacing, y_start),
        2,
    )
    pygame.draw.line(
        screen,
        white,
        (x_start, y_start + cell_height),
        (
            x_start + len(numbers) * (cell_width + spacing) - spacing,
            y_start + cell_height,
        ),
        2,
    )
    pygame.draw.line(
        screen,
        white,
        (x_start, y_start + cell_height * 2 + spacing),
        (
            x_start + len(numbers) * (cell_width + spacing) - spacing,
            y_start + cell_height * 2 + spacing,
        ),
        2,
    )
    pygame.draw.line(
        screen,
        white,
        (x_start, y_start),
        (x_start, y_start + cell_height * 2 + spacing),
        2,
    )


while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(black)

    # Check mouse press
    click, _, _ = pygame.mouse.get_pressed()
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None

    # Text predict
    title = largeFont.render("Predicted Text", True, white)
    screen.blit(title, (int(width / 7), 15))

    # Write probabilites
    draw_table(screen, numbers, probabilities)

    # Draw each grid cell
    cells = []
    for i in range(ROWS):
        row = []
        for j in range(COLS):
            rect = pygame.Rect(
                width / 2 + 15 + j * CELL_SIZE,
                15 + i * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )

            # If cell has been written on, darken cell
            if grid[i][j]:
                channel = 255 - (grid[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)

            # Draw blank cell
            else:
                pygame.draw.rect(screen, (211, 211, 211), rect)
            pygame.draw.rect(screen, black, rect, 1)

            # If writing on this cell, fill in current cell and neighbors
            if mouse and rect.collidepoint(mouse):
                grid[i][j] = 250 / 255
                if i + 1 < ROWS:
                    grid[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    grid[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    grid[i + 1][j + 1] = 190 / 255

    # predict button
    predictButton = pygame.Rect(80, 343 - 50, 100, 30)
    predictText = smallFont.render("Predict", True, white)
    predictTextRect = predictText.get_rect()
    predictTextRect.center = predictButton.center
    pygame.draw.rect(
        screen,
        white,
        (
            78,
            343 - 50 - 2,
            104,
            34,
        ),
        0,
        1,
    )
    pygame.draw.rect(screen, black, predictButton, 0, 1)

    screen.blit(predictText, predictTextRect)
    # Reset button
    resetButton = pygame.Rect(270, 343 - 50, 100, 30)
    resetText = smallFont.render("Reset", True, white)
    resetTextRect = resetText.get_rect()
    resetTextRect.center = resetButton.center
    pygame.draw.rect(
        screen,
        white,
        (
            268,
            343 - 50 - 2,
            104,
            34,
        ),
        0,
        1,
    )
    pygame.draw.rect(screen, black, resetButton)
    screen.blit(resetText, resetTextRect)

    if mouse and resetButton.collidepoint(mouse):
        grid = [[0] * COLS for _ in range(ROWS)]
        classified = None
        probabilities = [0] * 10

    if mouse and predictButton.collidepoint(mouse):

        screen.blit(predictText, predictTextRect)
        predictions = model.predict([np.array(grid).reshape(1, 28, 28, 1)])
        probabilities = [round(pred * 100, 2) for pred in predictions.tolist()[0]]
        classified = predictions.argmax()

    # Show classified
    if classified is not None:
        draw_table(screen, numbers, probabilities)
        classifiedText = extralargeFont.render(str(classified), True, white)
        classifiedRect = classifiedText.get_rect()
        grid_size = 20 * 2 + CELL_SIZE * COLS
        classifiedRect.center = ((width / 2) / 2, height / 2 - 50)
        screen.blit(classifiedText, classifiedRect)

    pygame.display.flip()
