from kivy.app import App
from kivy.uix.label import Label
from kivy.clock import Clock
from plyer import accelerometer

class OrientationDetectorApp(App):
    def build(self):
        self.label = Label(text='Orientation: hi')
        Clock.schedule_interval(self.update_orientation, 5.0 )  # Update every 0.1 seconds
        return self.label

    def update_orientation(self, dt):
        acceleration = accelerometer.acceleration[:3]
        if acceleration:
            x, y, z = acceleration
            orientation = self.get_orientation(x, y, z)
            self.label.text = f'Orientation: {orientation}'

    def get_orientation(self, x, y, z):
        # Adjust the threshold values based on your device's orientation sensitivity
        if abs(x) > abs(y) and abs(x) > abs(z):
            print("horizontal")
            return 'Horizontal'
        elif abs(y) > abs(x) and abs(y) > abs(z):
            print("vertical")
            return 'Vertical'
        else:
            print("unknown")
            return 'Unknown'

if __name__ == '__main__':
    for i in range(10):
        OrientationDetectorApp().run()
