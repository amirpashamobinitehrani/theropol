def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def canny(image, low_thresh=100, high_thresh=200):
  edges = cv2.Canny(image, low_thresh, high_thresh)
  return edges

def is_blurry(image, thresh=150):
  edges = cv2.Laplacian(image, cv2.CV_64F)
  return edges.var() < thresh

def sharpen(image, kernel):
  sharp = cv2.filter2D(resized_down, -1, kernel)
  return sharp
