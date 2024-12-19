import numpy as np
from PIL import Image, ImageDraw
import random


# As of now, only triangle has been implemented. 
class AbstractImageGenerator:
    def __init__(self, target_image, max_shapes=300, base_resolution=256):
        # Convert target image to RGB numpy array
        self.base_resolution = base_resolution
        self.target_image = np.array(target_image.convert('RGB').resize((base_resolution, base_resolution)), dtype=np.uint8)
        self.height, self.width, _ = self.target_image.shape
        
        # Initialize with average color of target image
        self.avg_color = tuple(np.mean(self.target_image, axis=(0,1)).astype(np.uint8))
        self.current_image = np.full_like(self.target_image, self.avg_color)
        
        self.max_shapes = max_shapes
        self.generated_shapes = []
    
    def root_mean_square_error(self, image1, image2):
        # Calculate Root Mean Square Error between two images
        diff = image1.astype(np.float32) - image2.astype(np.float32)
        return np.sqrt(np.mean(diff * diff))
    
    def generate_random_shape(self):
        
        color_r = np.random.randint(0, 256)
        color_g = np.random.randint(0, 256)
        color_b = np.random.randint(0, 256)
        
        # Generate triangle points
        points = [
            (np.random.randint(0, self.width), np.random.randint(0, self.height)),
            (np.random.randint(0, self.width), np.random.randint(0, self.height)),
            (np.random.randint(0, self.width), np.random.randint(0, self.height))
        ]
        
        return (
            points[0][0], points[0][1], # x1, y1
            points[1][0], points[1][1], # x2, y2
            points[2][0], points[2][1], # x3, y3
            color_r, color_g, color_b
        )
    
    def draw_shape(self, image, shape):
        
        x1, y1, x2, y2, x3, y3, r, g, b = shape
        
        
        output_image = image.copy()
        
       
        pil_image = Image.fromarray(output_image)
        draw = ImageDraw.Draw(pil_image)
        
        
        draw.polygon(
            [(x1, y1), (x2, y2), (x3, y3)], 
            fill=(r, g, b)
        )
        
        return np.array(pil_image)
    
    def mutate_shape(self, shape):
        
        x1, y1, x2, y2, x3, y3, r, g, b = shape
        
        # Vertex mutation
        mutation_type = np.random.randint(0, 3)
        if mutation_type == 0:
            # Mutate first vertex
            x1 = max(0, min(self.width-1, x1 + np.random.randint(-20, 21)))
            y1 = max(0, min(self.height-1, y1 + np.random.randint(-20, 21)))
        elif mutation_type == 1:
            # Mutate second vertex
            x2 = max(0, min(self.width-1, x2 + np.random.randint(-20, 21)))
            y2 = max(0, min(self.height-1, y2 + np.random.randint(-20, 21)))
        else:
            # Mutate third vertex
            x3 = max(0, min(self.width-1, x3 + np.random.randint(-20, 21)))
            y3 = max(0, min(self.height-1, y3 + np.random.randint(-20, 21)))
        
        # Color mutation
        color_mutation = np.random.randint(0, 3)
        if color_mutation == 0:
            r = max(0, min(255, r + np.random.randint(-30, 31)))
        elif color_mutation == 1:
            g = max(0, min(255, g + np.random.randint(-30, 31)))
        else:
            b = max(0, min(255, b + np.random.randint(-30, 31)))
        
        return (x1, y1, x2, y2, x3, y3, r, g, b)
    
    def hill_climbing(self):
        
        best_shape = None
        best_score = np.inf
        
        # Multiple attempts to find best initial shape
        for _ in range(20):
            current_shape = self.generate_random_shape()
            current_image = self.draw_shape(self.current_image, current_shape)
            current_score = self.root_mean_square_error(current_image, self.target_image)
            
            # Local search
            for _ in range(50):
                mutated_shape = self.mutate_shape(current_shape)
                mutated_image = self.draw_shape(self.current_image, mutated_shape)
                mutated_score = self.root_mean_square_error(mutated_image, self.target_image)
                
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
        
        for iteration in range(self.max_shapes):
            # Find best shape
            best_shape, score = self.hill_climbing()
            
            # Draw the best shape on current image
            self.current_image = self.draw_shape(self.current_image, best_shape)
            self.generated_shapes.append(best_shape)
            
            # Print progress
            print(f"Added shape {iteration + 1}, Score: {score}")
            
            # Optional: save intermediate result
            if (iteration + 1) % 20 == 0:
                Image.fromarray(self.current_image).save(f"output/progress_{iteration+1}.png")
        

        pil_image = Image.fromarray(self.current_image)
        
        resized_image = pil_image.resize(
            (target_resolution, target_resolution), 
            Image.BICUBIC
        )
        
        
        # sharpened_image = resized_image.filter(Image.SHARPEN)
        
        return resized_image

def main():
    # Target image
    target_image_path = "base_img/tree.png" 
    target_image = Image.open(target_image_path)
    
    # Create generator
    generator = AbstractImageGenerator(target_image, max_shapes=1200)
    
    # Generate abstract image with 512x512 resolution
    abstract_image = generator.generate_abstract_image(target_resolution=512)
    
    # Save results
    abstract_image.save("output/final.png")
    abstract_image.show()

if __name__ == "__main__":
    main()