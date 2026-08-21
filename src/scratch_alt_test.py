import io
import matplotlib.pyplot as plt
from IPython.display import display, Image

plt.figure()
plt.plot([1, 2], [3, 4])
buf = io.BytesIO()
plt.savefig(buf, format='png')
plt.close()

# display with alt text
display(Image(data=buf.getvalue(), alt="Test Alt Text"))
