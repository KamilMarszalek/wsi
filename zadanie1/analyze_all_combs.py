def analyze_all_combs(mass, mass_limit, price):
    bin_length = len(mass)
    max_price = 0
    max_mass = 0
    for i in range(2**bin_length):
        temp_mass = 0
        temp_value = 0
        bin_str = format(i, f"0{bin_length}b")
        for count, char in enumerate(bin_str):
            temp_mass += int(char) * mass[count]
            temp_value += int(char) * price[count]
        # print(bin_str, ':', temp_value)
        if temp_value > max_price and temp_mass <= mass_limit:
            max_price = temp_value
            max_mass = temp_mass
    return max_price, max_mass
