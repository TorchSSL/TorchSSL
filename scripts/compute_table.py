def compute_bar(low, up):
    fluctuation = (up - low) / 2
    avg = 10000 - (up + low) / 2
    print(str(round(avg * 0.01, 2)) + '+-' + str(round(fluctuation * 0.01, 2)))
    return avg, fluctuation


if __name__ == '__main__':
    while True:
        try:
            low, up = str(input('low and up value: ')).split(' ')
        except ValueError:
            break
        compute_bar(float(low), float(up))
