class BatchCalculatorContextManager:
    def __init__(self, filename):
        self.filename = filename

    def process_lines(self):
        with open(self.filename, 'r') as file:
            for line in file:
                result = self.procces_line(line.strip())
                print(f"Result for {line.strip()}: {result} ")

    def procces_line(self, line):
        try:
            result = eval(line)
            return result
        except Exception as e:
            return e

batch_calculator = BatchCalculatorContextManager('file.txt')
batch_calculator.process_lines()
