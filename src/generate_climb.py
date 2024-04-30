import cv2
import random
import utils
import argparse

def select_holds_for_problem(holds, num_holds=7, vertical_distance_percentage_range=(10, 20), horizontal_distance_percentage_range=(10, 20), max_start_height_percentage=30):
    if not holds or len(holds) < num_holds:
        raise ValueError(
            "Insufficient holds provided or not enough holds to create the problem.")

    # Sort by y-coordinate ascending, which means from top to bottom of the image
    sorted_holds = sorted(holds, key=lambda x: x[1])

    # Calculate the maximum allowable height for the start hold from the floor (lower part of the wall)
    # difference in highest and lowest y1 values
    total_height = sorted_holds[-1][1] - sorted_holds[0][1]
    max_start_height = sorted_holds[-1][1] - \
        total_height * (max_start_height_percentage / 100)
    
    # Filter possible starting holds based on the max allowable height
    possible_starts = [h for h in sorted_holds if h[1] >= max_start_height]
    if not possible_starts:
        raise ValueError(
            "No suitable starting holds found close to the floor.")

    # Randomly select a start hold from those close to the floor
    start_hold = random.choice(possible_starts)

    # Calculate total wall width for horizontal movement
    total_width = max(hold[2] for hold in holds) - \
        min(hold[0] for hold in holds)

    # Convert percentage ranges to actual distances for vertical and horizontal distances
    min_vertical_distance = total_height * \
        (vertical_distance_percentage_range[0] / 100)
    max_vertical_distance = total_height * \
        (vertical_distance_percentage_range[1] / 100)
    min_horizontal_distance = total_width * \
        (horizontal_distance_percentage_range[0] / 100)
    max_horizontal_distance = total_width * \
        (horizontal_distance_percentage_range[1] / 100)

    next_holds = []
    current_hold = start_hold

    while len(next_holds) < num_holds - 1:

        # Filter holds based on vertical and horizontal distances from the current hold, ensuring each is higher than the last
        possible_holds = [
            h for h in sorted_holds
            # Ensure the hold is higher than the current hold (lower y1 value)
            if h[1] < current_hold[1] and
            abs(current_hold[1] - h[1]) >= min_vertical_distance and
            abs(current_hold[1] - h[1]) <= max_vertical_distance and
            abs(current_hold[0] - h[0]) >= min_horizontal_distance and
            abs(current_hold[0] - h[0]) <= max_horizontal_distance
        ]

        if not possible_holds:
            break  # Stop if no holds meet the criteria within the maximum distances

        chosen_hold = random.choice(possible_holds)
        next_holds.append(chosen_hold)
        current_hold = chosen_hold  # Update the current hold to the new chosen hold

    problem_holds = [start_hold] + next_holds 
    return problem_holds




def visualize_problem(image, holds):

    red = (0, 0, 255) #intermediate hold color
    green = (0, 255, 0) #start and end hold color

    # Draw colored boxes around problem
    for i in range(len(holds)):
        x1, y1, x2, y2 = holds[i]
        color = None
        if i == 2 or i == len(holds)-1:
            color = green
        else:
            color = red
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    cv2.imshow('Generated climb', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_climb(args):
    image = cv2.imread(args.img_path)
    detected_boxes = utils.get_all_boxes(img_path=args.img_path)

    problem_holds = select_holds_for_problem(
        holds=detected_boxes,
        num_holds=args.num_holds,
        vertical_distance_percentage_range=(
            args.vertical_min, args.vertical_max),
        horizontal_distance_percentage_range=(
            args.horizontal_min, args.horizontal_max),
        max_start_height_percentage=args.max_start_height
    )

    masked = None

    if args.show:
        visualize_problem(image, problem_holds)
        masked = utils.only_show_valid_holds(img=image, valid_boxes=problem_holds,show=True)

    if args.save:
        img_name = args.img_path.split('/')[1].split('.')[0]
        if masked is None:
            masked = utils.only_show_valid_holds(
                img=image, valid_boxes=problem_holds, show=False)

        cv2.imwrite(f'output/generated_climb/{img_name}_generated.png', image)
        cv2.imwrite(f'output/generated_climb/{img_name}_generated_masked.png', masked)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate climbing routes from images.")
    parser.add_argument("--img_path", type=str,
                        default='input/test_image_2.png', help="Path to the input image")
    parser.add_argument("--num_holds", type=int, default=7,
                        help="Number of holds to generate in the problem")
    parser.add_argument("--vertical_min", type=int, default=5,
                        help="Minimum vertical distance percentage")
    parser.add_argument("--vertical_max", type=int, default=20,
                        help="Maximum vertical distance percentage")
    parser.add_argument("--horizontal_min", type=int, default=5,
                        help="Minimum horizontal distance percentage")
    parser.add_argument("--horizontal_max", type=int, default=20,
                        help="Maximum horizontal distance percentage")
    parser.add_argument("--max_start_height", type=int, default=30,
                        help="Maximum start height percentage from the bottom")
    parser.add_argument("--show", action='store_true',
                        help="Show the generated climb")
    parser.add_argument("--save", action='store_true',
                        help="Save the generated climb image")
    args = parser.parse_args()

    generate_climb(args)

