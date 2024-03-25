import tkinter
from tkinter import *
from tkinter import filedialog, ttk, messagebox

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_path = ""
menu = tkinter.Tk()

font_type = "Verdana"
font_size = 12
frame_font_size = 14

selected_offense = StringVar()
selected_offense.set("NONE")
selected_offense_tree = StringVar()
selected_offense_tree.set("NONE")


def quit_system():
    menu.destroy()


def open_cluster_window():
    menu.withdraw()

    # function

    def close_cluster():
        cluster_window.destroy()
        menu.deiconify()
        return None

    def quit_cluster():
        cluster_window.destroy()
        menu.destroy()
        return None

    def file_dialog():
        filename = filedialog.askopenfilename(initialdir="/", title="Select CSV File",
                                              filetype=(("csv files", "*.csv"), ("All Files", "*.*")))
        label_file["text"] = filename
        return None

    def load_excel_data():
        file_path_directory = label_file["text"]
        try:
            filename = r"{}".format(file_path_directory)
            if filename[-4:] == ".csv":
                df = pd.read_csv(filename)
            else:
                messagebox.showerror("Error", "Non-CSV file detected. Please load CSV file")
                return None
            df = df.dropna()
            df.reset_index(drop=True)
            messagebox.showinfo("Information", "CSV File loaded successfully")
        except FileNotFoundError:
            messagebox.showerror("Error", "No CSV file found.")
            return None

        clear_data()
        dataset_tree_view["column"] = list(df.columns)
        dataset_tree_view["show"] = "headings"
        for column in dataset_tree_view["columns"]:
            dataset_tree_view.heading(column, text=column)

        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            dataset_tree_view.insert("", "end", values=row)

        offense_unique_list = list(df.iloc[:, 0].unique())
        offense_unique_list = sorted(offense_unique_list)

        offense_dropdown = OptionMenu(setting_frame, selected_offense, *offense_unique_list, command=get_offense)
        offense_dropdown.config(font=(font_type, font_size), borderwidth=4)
        offense_dropdown.place(x=10, y=15)
        return None

    def clear_data():
        dataset_tree_view.delete(*dataset_tree_view.get_children())
        return None

    def get_feature1(val1):
        feature1_selected.configure(text=val1)

    def get_feature2(val2):
        feature2_selected.configure(text=val2)

    def get_num_cluster(val3):
        cluster_selected.configure(text=val3)

    def get_offense(val4):
        offense_selected.configure(text=val4)

    def predict_model():
        f1 = feature1_selected.cget("text")
        f2 = feature2_selected.cget("text")
        clu = int(float(cluster_selected.cget("text")))
        off = offense_selected.cget("text")

        try:
            city = pd.read_csv(label_file["text"])
        except:
            messagebox.showerror("Error", "No CSV file is found. Please select a CSV file.")
            return None

        filter_city = pd.DataFrame(city)
        filter_city = filter_city.dropna()
        filter_city.reset_index(drop=False)

        if off != "None":
            offense_filter = pd.DataFrame(filter_city.loc[filter_city["Offense_Description"] == off])
            offense_filter.reset_index(drop=False)
        elif off == "None":
            messagebox.showerror("Error", "Offense Selection is not selected. Please select one.")
            return None

        if f1 == f2:
            messagebox.showerror("Error", "Both features are the same. Please select different features for each both.")
            return None

        x_column, y_column = f1, f2
        offense_feature_filter = offense_filter[[x_column, y_column]]

        row_count, previous_row_count = len(offense_feature_filter.index), 1
        while row_count != previous_row_count:
            previous_row_count = len(offense_feature_filter.index)
            z = np.abs(stats.zscore(offense_feature_filter))
            offense_feature_filter = offense_feature_filter[(z < 3).all(axis=1)]
            row_count = len(offense_feature_filter.index)

        k_means = KMeans(n_clusters=clu, n_init=30, max_iter=200, tol=0.001, algorithm='elkan').fit(
            offense_feature_filter)
        centroids = k_means.cluster_centers_

        plt.scatter(offense_feature_filter[x_column], offense_feature_filter[y_column], c=k_means.labels_.astype(float),
                    s=70, alpha=0.9)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=250, marker='x')

        x_label = '%s' % x_column
        y_label = '%s' % y_column
        crime_label = '%s' % off
        title = '%s - %s and %s' % (crime_label, x_label, y_label)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.rcParams["figure.figsize"] = [12, 12]
        plt.show()

        sum_of_squared_error = []

        for number_cluster in range(1, 20):
            k_means = KMeans(n_clusters=number_cluster, init='k-means++')
            k_means.fit(offense_feature_filter)
            sum_of_squared_error.append(k_means.inertia_)

        frame = pd.DataFrame({'Cluster': range(1, 20), 'SSE': sum_of_squared_error})
        plt.figure(figsize=(8, 6))
        plt.plot(frame['Cluster'], frame['SSE'], marker='o')
        plt.title('Evaluation Metrics - %s' % crime_label)
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of Squared Errors')
        plt.show()
        return None

    # Windows Initialization
    cluster_window = Tk()
    cluster_window.title("Predicting Cluster Between Two Factors")
    cluster_window.geometry("780x900+50+50")
    cluster_window.resizable(False, False)
    cluster_window.protocol('WM_DELETE_WINDOW', (lambda: 'pass')())
    cluster_window.resizable(0, 0)

    # LabelFrame
    dataset_frame = LabelFrame(cluster_window, text="Dataset", width=750, height=400, font=(font_type,
                                                                                            frame_font_size))
    dataset_frame.place(x=10, y=5)
    load_csv_frame = LabelFrame(cluster_window, text="Load CSV File", width=750, height=130, font=(font_type,
                                                                                                   frame_font_size))
    load_csv_frame.place(x=10, y=420)
    setting_frame = LabelFrame(cluster_window, text="Setting", width=750, height=260, font=(font_type,
                                                                                            frame_font_size))
    setting_frame.place(x=10, y=570)
    tree_view_frame = LabelFrame(dataset_frame)
    tree_view_frame.place(x=10, y=5, width=720, height=360)

    # UI Element
    load_file_button = Button(load_csv_frame, text="Load CSV File", width=12, height=2, borderwidth=4,
                              font=(font_type, font_size), command=load_excel_data)
    load_file_button.place(x=400, y=45)
    load_csv_button = Button(load_csv_frame, text="Browse CSV File", width=14, height=2, borderwidth=4,
                             font=(font_type, font_size), command=file_dialog)
    load_csv_button.place(x=200, y=45)

    dataset_tree_view = ttk.Treeview(tree_view_frame)
    dataset_tree_view.place(relheight=1, relwidth=1)

    tree_scroll_y = tkinter.Scrollbar(tree_view_frame, orient="vertical", command=dataset_tree_view.yview)
    tree_scroll_x = tkinter.Scrollbar(tree_view_frame, orient="horizontal", command=dataset_tree_view.xview)
    dataset_tree_view.configure(xscrollcommand=tree_scroll_x.set, yscrollcommand=tree_scroll_y.set)
    tree_scroll_x.pack(side="bottom", fill="x")
    tree_scroll_y.pack(side="right", fill="y")

    label_path = Label(load_csv_frame, text="Filepath: ", font=(font_type, font_size))
    label_path.place(y=10, x=10)
    label_file = Label(load_csv_frame, text="", font=(font_type, font_size))
    label_file.place(y=10, x=88)

    feature1_label = Label(setting_frame, text="Factor 1:", font=(font_type, font_size))
    feature2_label = Label(setting_frame, text="Factor 2:", font=(font_type, font_size))
    feature1 = StringVar(cluster_window)
    feature2 = StringVar(cluster_window)
    feature1.set("Hour")
    feature2.set("Hour")
    feature1_dropdown = OptionMenu(setting_frame, feature1, "Month", "Day Of Month", "Hour", "Minute",
                                   command=get_feature1)
    feature1_dropdown.config(font=(font_type, font_size), borderwidth=4)
    feature2_dropdown = OptionMenu(setting_frame, feature2, "Month", "Day Of Month", "Hour", "Minute",
                                   command=get_feature2)
    feature2_dropdown.config(font=(font_type, font_size), borderwidth=4)

    feature1_label.place(x=30, y=60)
    feature2_label.place(x=30, y=100)
    feature1_dropdown.place(x=120, y=55)
    feature2_dropdown.place(x=120, y=95)

    num_cluster_label = Label(setting_frame, text="Number of Cluster: ", font=(font_type, font_size))
    num_cluster = IntVar(cluster_window)
    num_cluster.set(1)
    num_cluster_dropdown = OptionMenu(setting_frame, num_cluster, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                      17, 18, 19, 20, command=get_num_cluster)
    num_cluster_dropdown.config(font=(font_type, font_size), borderwidth=4)
    num_cluster_label.place(x=10, y=170)
    num_cluster_dropdown.place(x=180, y=165)

    offense_label = Label(setting_frame, text="Offense Description: ", font=(font_type, font_size))
    offense_label.place(x=60, y=20)

    feature1_selected = Label(setting_frame, text="Hour", font=(font_type, font_size))
    feature2_selected = Label(setting_frame, text="Hour", font=(font_type, font_size))
    cluster_selected = Label(setting_frame, text="1", font=(font_type, font_size))
    offense_selected = Label(setting_frame, text="None", font=(font_type, font_size))
    feature1_selected.place(x=600, y=600)
    feature2_selected.place(x=600, y=600)
    cluster_selected.place(x=600, y=600)
    offense_selected.place(x=230, y=20)

    predict_button = tkinter.Button(setting_frame, text="Predict", height=2, width=10, borderwidth=8,
                                    font=(font_type, font_size), command=predict_model)
    predict_button.place(x=400, y=130)
    back_button = tkinter.Button(cluster_window, text="Close", height=2, width=14, borderwidth=4,
                                 font=(font_type, font_size), command=close_cluster)
    back_button.place(x=220, y=840)

    quit_cluster_button = tkinter.Button(cluster_window, text="Quit", height=2, width=14, borderwidth=4,
                                         font=(font_type, font_size), command=quit_cluster)
    quit_cluster_button.place(x=380, y=840)


def open_location_window():
    menu.withdraw()

    # function
    def close_tree():
        location_window.destroy()
        menu.deiconify()
        return None

    def quit_tree():
        location_window.destroy()
        menu.destroy()
        return None

    def file_dialog_tree():
        filename = filedialog.askopenfilename(initialdir="/", title="Select CSV File",
                                              filetype=(("csv files", "*.csv"), ("All Files", "*.*")))
        label_file_tree["text"] = filename
        return None

    def load_excel_data_tree():
        file_path_directory = label_file_tree["text"]
        try:
            filename = r"{}".format(file_path_directory)
            if filename[-4:] == ".csv":
                df_tree = pd.read_csv(filename)
            else:
                messagebox.showerror("Error", "Non-CSV file detected. Please load CSV file")
                return None
            df_tree = df_tree.dropna()
            df_tree.reset_index(drop=True)
            messagebox.showinfo("Information", "CSV File loaded successfully")
        except FileNotFoundError:
            messagebox.showerror("Error", "No CSV file found.")
            return None

        clear_data_tree()
        dataset_tree_view_tree["column"] = list(df_tree.columns)
        dataset_tree_view_tree["show"] = "headings"
        for column in dataset_tree_view_tree["columns"]:
            dataset_tree_view_tree.heading(column, text=column)

        df_rows_tree = df_tree.to_numpy().tolist()
        for row in df_rows_tree:
            dataset_tree_view_tree.insert("", "end", values=row)

        offense_unique_list = list(df_tree.iloc[:, 0].unique())
        offense_unique_list = sorted(offense_unique_list)

        offense_dropdown = OptionMenu(setting_frame_tree, selected_offense_tree, *offense_unique_list,
                                      command=select_offense)
        offense_dropdown.config(font=(font_type, font_size), borderwidth=4)
        offense_dropdown.place(x=20, y=135)
        return None

    def select_offense(value):
        selected_offense_text.configure(text=value)

    def clear_data_tree():
        dataset_tree_view_tree.delete(*dataset_tree_view_tree.get_children())
        return None

    def get_month(month):
        month_selected_text.configure(text=month)

    def get_day_of_month(day_of_month):
        day_of_month_selected_text.configure(text=day_of_month)

    def get_hour(hour):
        hour_selected_text.configure(text=hour)

    def get_minute(minute):
        minute_selected_text.configure(text=minute)

    def select_algorithm(algorithm):
        algorithm_selected_text.configure(text=algorithm)

    def train_model():

        month_selected = int(float(month_selected_text.cget("text")))
        day_of_month_selected = int(float(day_of_month_selected_text.cget("text")))
        hour_selected = int(float(hour_selected_text.cget("text")))
        minute_selected = int(float(minute_selected_text.cget("text")))
        offense_selected = selected_offense_text.cget("text")
        algorithm_selected = algorithm_selected_text.cget("text")

        try:
            city = pd.read_csv(label_file_tree["text"])
            city = city.dropna()
        except:
            messagebox.showerror("Error", "No CSV file is found. Please select a CSV file.")
            return None

        filter_city = pd.DataFrame(city)
        filter_city = filter_city.dropna()
        filter_city.reset_index(drop=False)

        if offense_selected != "None":
            offense_filter_tree = pd.DataFrame(filter_city.loc[filter_city["Offense_Description"] == offense_selected])
            offense_filter_tree.reset_index(drop=False)
        elif offense_selected == "None":
            messagebox.showerror("Error", "Offense Selection is not selected. Please select one.")
            return None

        x = offense_filter_tree
        x = x.drop(columns=['Offense_Description', 'Year', 'Street'])

        try:
            x = x.drop(columns=['Occurred_On_Date'])
        except:
            print("No Occurred Date feature.")

        y = offense_filter_tree.Street

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

        if algorithm_selected == "Decision Tree":
            clf = DecisionTreeClassifier(criterion='entropy')
        elif algorithm_selected == "Random Forest":
            clf = RandomForestClassifier(n_estimators=50)
        elif algorithm_selected == "Naïve Bayes":
            clf = GaussianNB()
        clf = clf.fit(x_train, y_train)
        y_prediction = clf.predict(x_test)

        accuracy = accuracy_score(y_test, y_prediction)
        accuracy_number.configure(text=(str(round(accuracy*100, 2))+"%"))
        messagebox.showinfo("Prediction", "Finished Prediction")

        predicted_street = clf.predict([[month_selected, day_of_month_selected, hour_selected, minute_selected,0,0]])
        predicted_result = np.array2string(predicted_street)
        predicted_result = predicted_result.replace("['", "")
        predicted_result = predicted_result.replace("']", "")
        predicted_result = predicted_result.replace("/", " ")
        predict_street_label.configure(text=predicted_result)
        return None

    # Windows Initialization
    location_window = Tk()
    location_window.title("Predicting Location ")
    location_window.geometry("830x980+5+5")
    location_window.resizable(False, False)
    location_window.protocol('WM_DELETE_WINDOW', (lambda: 'pass')())

    # LabelFrame
    dataset_frame_tree = LabelFrame(location_window, text="Dataset", width=800, height=400, font=(font_type,
                                                                                                  frame_font_size))
    dataset_frame_tree.place(x=10, y=5)
    load_csv_frame_tree = LabelFrame(location_window, text="Load CSV File", width=800, height=150,
                                     font=(font_type, frame_font_size))
    load_csv_frame_tree.place(x=10, y=420)
    setting_frame_tree = LabelFrame(location_window, text="Setting", width=800, height=230,
                                    font=(font_type, frame_font_size))
    setting_frame_tree.place(x=10, y=580)
    metric_frame_tree = LabelFrame(location_window, text="Metric & Result", width=390, height=120, font=(font_type,
                                                                                                    frame_font_size))
    metric_frame_tree.place(x=420, y=820)
    tree_view_frame_tree = LabelFrame(dataset_frame_tree)
    tree_view_frame_tree.place(x=10, y=10, width=770, height=350)

    # UI Element
    load_file_button_tree = Button(load_csv_frame_tree, text="Load CSV File", width=12, height=2, borderwidth=4,
                                   font=(font_type, font_size), command=load_excel_data_tree)
    load_file_button_tree.place(x=400, y=60)
    load_csv_button_tree = Button(load_csv_frame_tree, text="Browse CSV File", width=14, height=2, borderwidth=4,
                                  font=(font_type, font_size), command=file_dialog_tree)
    load_csv_button_tree.place(x=200, y=60)

    dataset_tree_view_tree = ttk.Treeview(tree_view_frame_tree)
    dataset_tree_view_tree.place(relheight=1, relwidth=1)

    tree_scroll_y = tkinter.Scrollbar(tree_view_frame_tree, orient="vertical", command=dataset_tree_view_tree.yview)
    tree_scroll_x = tkinter.Scrollbar(tree_view_frame_tree, orient="horizontal", command=dataset_tree_view_tree.xview)
    dataset_tree_view_tree.configure(xscrollcommand=tree_scroll_x.set, yscrollcommand=tree_scroll_y.set)
    tree_scroll_x.pack(side="bottom", fill="x")
    tree_scroll_y.pack(side="right", fill="y")

    label_path_tree = Label(load_csv_frame_tree, text="Filepath: ", font=(font_type, font_size))
    label_path_tree.place(y=10, x=10)
    label_file_tree = Label(load_csv_frame_tree, text="", font=(font_type, font_size))
    label_file_tree.place(y=10, x=90)

    offense_text = Label(setting_frame_tree, text="Selected Offense:", font=(font_type, font_size))
    offense_text.place(x=70, y=140)
    selected_offense_text = Label(setting_frame_tree, text="None", font=(font_type, font_size))
    selected_offense_text.place(x=220, y=140)

    accuracy_text = Label(metric_frame_tree, text="Accuracy:", font=(font_type, font_size))
    accuracy_text.place(x=10, y=45)
    accuracy_number = Label(metric_frame_tree, text="0%", font=(font_type, font_size))
    accuracy_number.place(x=90, y=45)

    month_text = Label(setting_frame_tree, text="Month: ", font=(font_type, font_size))
    month_text.place(x=120, y=30)
    day_of_month_text = Label(setting_frame_tree, text="Day of Month: ", font=(font_type, font_size))
    day_of_month_text.place(x=60, y=70)
    hour_text = Label(setting_frame_tree, text="Hour: ", font=(font_type, font_size))
    hour_text.place(x=280, y=30)
    minute_text = Label(setting_frame_tree, text="Minute: ", font=(font_type, font_size))
    minute_text.place(x=264, y=70)
    algorithm_street_text = Label(setting_frame_tree, text="Algorithm:",  font=(font_type, font_size))
    algorithm_street_text.place(x=460, y=100)
    predict_street_text = Label(metric_frame_tree, text="Predicted Street:", font=(font_type, font_size))
    predict_street_text.place(x=10, y=20)

    month_feature = IntVar(location_window)
    month_feature.set(1)
    day_of_month_feature = IntVar(location_window)
    day_of_month_feature.set(1)
    hour_feature = IntVar(location_window)
    hour_feature.set(0)
    minute_feature = IntVar(location_window)
    minute_feature.set(0)
    algorithm_selected = StringVar(location_window)
    algorithm_selected.set("Decision Tree")

    month_dropdown = OptionMenu(setting_frame_tree, month_feature, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                command=get_month)
    month_dropdown.config(font=(font_type, font_size), borderwidth=4)
    month_dropdown.place(x=190, y=23)

    day_of_month_dropdown = OptionMenu(setting_frame_tree, day_of_month_feature, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                       command=get_day_of_month)
    day_of_month_dropdown.config(font=(font_type, font_size), borderwidth=4)
    day_of_month_dropdown.place(x=190, y=63)

    hour_dropdown = OptionMenu(setting_frame_tree, hour_feature, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                               16, 17, 18, 19, 20, 21, 22, 23, command=get_hour)
    hour_dropdown.config(font=(font_type, font_size), borderwidth=4)
    hour_dropdown.place(x=340, y=23)

    minute_dropdown = OptionMenu(setting_frame_tree, minute_feature, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
                                 command=get_minute)
    minute_dropdown.config(font=(font_type, font_size), borderwidth=4)
    minute_dropdown.place(x=340, y=63)

    algorithm_dropdown = OptionMenu(setting_frame_tree, algorithm_selected, "Decision Tree", "Random Forest",
                                    "Naïve Bayes", command=select_algorithm)
    algorithm_dropdown.config(font=(font_type, font_size), borderwidth=4)
    algorithm_dropdown.place(x=550, y=95)

    predict_street_label = Label(metric_frame_tree, text="None", font=(font_type, font_size))
    predict_street_label.place(x=150, y=20)

    predict_button_tree = tkinter.Button(setting_frame_tree, text="Predict", height=2, width=10, borderwidth=8,
                                         command=train_model, font=(font_type, font_size))
    predict_button_tree.place(x=550, y=20)
    back_button_tree = tkinter.Button(location_window, text="Close", height=2, width=14, borderwidth=4,
                                      command=close_tree, font=(font_type, font_size))
    back_button_tree.place(x=50, y=850)
    quit_button_tree = tkinter.Button(location_window, text="Quit", height=2, width=14, borderwidth=4,
                                      command=quit_tree, font=(font_type, font_size))
    quit_button_tree.place(x=220, y=850)

    month_selected_text = Label(setting_frame_tree, text="1", font=(font_type, font_size))
    month_selected_text.place(x=400, y=400)
    day_of_month_selected_text = Label(setting_frame_tree, text="1", font=(font_type, font_size))
    day_of_month_selected_text.place(x=400, y=400)
    hour_selected_text = Label(setting_frame_tree, text="0", font=(font_type, font_size))
    hour_selected_text.place(x=400, y=400)
    minute_selected_text = Label(setting_frame_tree, text="0", font=(font_type, font_size))
    minute_selected_text.place(x=400, y=400)
    algorithm_selected_text = Label(setting_frame_tree, text="Decision Tree", font=(font_type, font_size))
    algorithm_selected_text.place(x=400, y=400)


menu.title("Predictive Policing System")
menu.geometry("800x400+100+100")
menu.resizable(False, False)
menu.protocol('WM_DELETE_WINDOW', (lambda: 'pass')())

# Question Label
question = Label(menu, text="Crime Prediction System")
question.config(font=(font_type, 14))
question.place(x=280, y=30)

# Button to Clustering and Location
cluster = tkinter.Button(menu, text="Clustering Between Two Features", height=4, width=32, borderwidth=8,
                         font=(font_type, font_size), command=open_cluster_window)
location_tree = tkinter.Button(menu, text="Finding The Predicted Location", height=4, width=28,
                               borderwidth=8, font=(font_type, font_size), command=open_location_window)
quit_button = tkinter.Button(menu, text="Quit", height=2, width=14, borderwidth=2, font=(font_type, font_size),
                             command=quit_system)

cluster.place(x=50, y=150)
location_tree.place(x=450, y=150)
quit_button.place(x=340, y=300)

menu.mainloop()
