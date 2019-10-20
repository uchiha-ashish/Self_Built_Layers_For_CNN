model = DeceptiNet()
model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              steps_per_epoch = 772,
                              epochs = 50,
                              validation_data = validation_generator,
                              validation_steps = 97,
                              verbose = 1)
                              
model.evaluate(x_test, y_test)
