import numpy as np
import os
from PIL import Image, ImageDraw
import argparse
import time


class ShapeImageGenerator:
    def __init__(self, target_image, shape='circle', count=300, base_resolution=256, output_directory="output", keep_progress=True):
        self.shape_type = shape.lower()
        if self.shape_type not in ['circle', 'triangle']:
            raise ValueError(
                "Shape shape must be either 'circle' or 'triangle'")

        self.output_directory = output_directory
        self.keep_progress = keep_progress

        # Convert and resize target image
        self.base_resolution = base_resolution
        self.target_image = np.array(target_image.convert('RGB').resize(
            (base_resolution, base_resolution)), dtype=np.uint8)
        self.height, self.width, _ = self.target_image.shape

        # Initialize with average color
        self.avg_color = tuple(
            np.mean(self.target_image, axis=(0, 1)).astype(np.uint8))
        self.current_image = np.full_like(self.target_image, self.avg_color)

        self.max_shapes = count
        self.generated_shapes = []

        # Use numpy's random number generator for better performance
        self.rng = np.random.default_rng()

    def root_mean_square_error(self, image1, image2):
        diff = image1.astype(np.float32) - image2.astype(np.float32)
        return np.sqrt(np.mean(diff * diff))

    def generate_random_shape(self):
        if self.shape_type == 'circle':
            center_x = self.rng.integers(0, self.width)
            center_y = self.rng.integers(0, self.height)
            radius = self.rng.integers(1, self.width // 4)
            colors = self.rng.integers(0, 256, 3)
            return (center_x, center_y, radius, *colors)
        else:  # triangle
            points = self.rng.integers(
                0, [self.width, self.height], size=(3, 2))
            colors = self.rng.integers(0, 256, 3)
            return (*points.flatten(), *colors)

    def draw_shape(self, image, shape):
        output_image = image.copy()
        pil_image = Image.fromarray(output_image)
        draw = ImageDraw.Draw(pil_image)

        if self.shape_type == 'circle':
            center_x, center_y, radius, r, g, b = shape
            left_up = (center_x - radius, center_y - radius)
            right_down = (center_x + radius, center_y + radius)
            draw.ellipse([left_up, right_down], fill=(r, g, b))
        else:  # triangle
            x1, y1, x2, y2, x3, y3, r, g, b = shape
            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=(r, g, b))

        return np.array(pil_image)

    def mutate_shape(self, shape):
        if self.shape_type == 'circle':
            center_x, center_y, radius, r, g, b = shape

            mutation_type = self.rng.integers(0, 4)
            if mutation_type == 0:
                center_x = max(0, min(self.width-1, center_x +
                               self.rng.integers(-20, 21)))
            elif mutation_type == 1:
                center_y = max(
                    0, min(self.height-1, center_y + self.rng.integers(-20, 21)))
            elif mutation_type == 2:
                radius = max(1, min(self.width//2, radius +
                             self.rng.integers(-10, 11)))
            else:
                color_idx = self.rng.integers(0, 3)
                colors = [r, g, b]
                colors[color_idx] = max(
                    0, min(255, colors[color_idx] + self.rng.integers(-30, 31)))
                r, g, b = colors

            return (center_x, center_y, radius, r, g, b)
        else:  # triangle
            x1, y1, x2, y2, x3, y3, r, g, b = shape

            vertex = self.rng.integers(0, 3)
            if vertex == 0:
                x1 = max(0, min(self.width-1, x1 + self.rng.integers(-20, 21)))
                y1 = max(0, min(self.height-1, y1 + self.rng.integers(-20, 21)))
            elif vertex == 1:
                x2 = max(0, min(self.width-1, x2 + self.rng.integers(-20, 21)))
                y2 = max(0, min(self.height-1, y2 + self.rng.integers(-20, 21)))
            else:
                x3 = max(0, min(self.width-1, x3 + self.rng.integers(-20, 21)))
                y3 = max(0, min(self.height-1, y3 + self.rng.integers(-20, 21)))

            if self.rng.random() < 0.3:  # 30% chance to mutate color
                color_idx = self.rng.integers(0, 3)
                colors = [r, g, b]
                colors[color_idx] = max(
                    0, min(255, colors[color_idx] + self.rng.integers(-30, 31)))
                r, g, b = colors

            return (x1, y1, x2, y2, x3, y3, r, g, b)

    def hill_climbing(self):
        best_shape = None
        best_score = np.inf

        # Multiple attempts to find best initial shape
        for _ in range(20):
            current_shape = self.generate_random_shape()
            current_image = self.draw_shape(self.current_image, current_shape)
            current_score = self.root_mean_square_error(
                current_image, self.target_image)

            # Local search
            for _ in range(50):
                mutated_shape = self.mutate_shape(current_shape)
                mutated_image = self.draw_shape(
                    self.current_image, mutated_shape)
                mutated_score = self.root_mean_square_error(
                    mutated_image, self.target_image)

                if mutated_score < current_score:
                    current_shape = mutated_shape
                    current_image = mutated_image
                    current_score = mutated_score

                if current_score < 10:
                    break

            if current_score < best_score:
                best_shape = current_shape
                best_score = current_score

        return best_shape, best_score

    def generate_abstract_image(self, target_resolution=512):
        start_time = time.time()

        for iteration in range(self.max_shapes):
            # Find best shape
            best_shape, score = self.hill_climbing()

            # Draw the best shape on current image
            self.current_image = self.draw_shape(
                self.current_image, best_shape)
            self.generated_shapes.append(best_shape)

            # Print progress with time estimate
            elapsed_time = time.time() - start_time
            shapes_per_second = (iteration + 1) / elapsed_time
            remaining_shapes = self.max_shapes - (iteration + 1)
            estimated_remaining_time = remaining_shapes / shapes_per_second

            print(f"Added shape {iteration +
                  1}/{self.max_shapes}, Score: {score:.2f}")
            print(f"Elapsed: {elapsed_time:.1f}s, Estimated remaining: {
                  estimated_remaining_time:.1f}s")

            # Save intermediate result
            if self.keep_progress and (iteration + 1) % 20 == 0:
                Image.fromarray(self.current_image).save(
                    f"{self.output_directory}/progress_{iteration+1}.png")

        pil_image = Image.fromarray(self.current_image)
        resized_image = pil_image.resize(
            (target_resolution, target_resolution),
            Image.BICUBIC
        )

        print(f"\nTotal time: {time.time() - start_time:.1f} seconds")
        return resized_image


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate abstract image using geometric shapes')

    # Required arguments
    parser.add_argument('-i', '--image', required=True,
                        help='Path to input image')

    # Optional arguments with shorter aliases
    parser.add_argument('-s', '--shape', choices=['circle', 'triangle'], default='circle',
                        help='Geometric shape to use (default: circle)')
    parser.add_argument('-n', '--number', type=int, default=1200,
                        help='Number of shapes to generate (default: 1200)')
    parser.add_argument('-r', '--resolution', type=int, default=512,
                        help='Output resolution (default: 512)')
    parser.add_argument('-w', '--working-res', type=int, default=256,
                        help='Working resolution (default: 256)')
    parser.add_argument('-k', '--keep-progress', action='store_true', default=True,
                        help='Save progress images every 20 iterations (default: True)')
    parser.add_argument('--no-progress', action='store_false', dest='keep_progress',
                        help='Disable saving progress images')

    args = parser.parse_args()

    # Load target image
    target_image = Image.open(args.image)

    # Output Directory
    directory = "output"
    if args.keep_progress and not os.path.exists(directory):
        os.makedirs(directory)

    # Create generator
    generator = ShapeImageGenerator(
        target_image,
        shape=args.shape,
        count=args.number,
        base_resolution=args.working_res,
        output_directory=directory,
        keep_progress=args.keep_progress
    )

    # Generate abstract image
    abstract_image = generator.generate_abstract_image(
        target_resolution=args.resolution)

    # Save results
    output_filename = f"output/final_{
        args.shape}s.png" if args.keep_progress else f"final_{args.shape}s.png"
    abstract_image.save(output_filename)
    print(f"\nSaved final image to: {output_filename}")
    abstract_image.show()


if __name__ == "__main__":
    main()
